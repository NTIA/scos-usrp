"""Maintains a persistent connection to the USRP.

Example usage:
    >>> from scos_usrp.hardware import radio
    >>> radio.is_available
    True
    >>> rx = radio
    >>> rx.sample_rate = 10e6
    >>> rx.frequency = 700e6
    >>> rx.gain = 40
    >>> samples = rx.acquire_time_domain_samples(1000)
"""

import logging
import time
from datetime import datetime, timedelta

import numpy as np

# from hardware.radio_iface import RadioInterface
from scos_actions import utils
from scos_actions.hardware.radio_iface import RadioInterface

from scos_usrp import settings
from scos_usrp.hardware import calibration
from scos_usrp.hardware.mocks.usrp_block import MockUsrp
from scos_usrp.hardware.tests.resources.utils import create_dummy_calibration

logger = logging.getLogger(__name__)

# Testing determined these gain values provide a good mix of sensitivity and
# dynamic range performance
VALID_GAINS = (0, 20, 40, 60)

CLOCK_TIMEOUT = 1000  # 1000mS timeout for external clock locking

# Define the default calibration dicts
DEFAULT_SIGAN_CALIBRATION = {
    "gain_sigan": None,  # Defaults to gain setting
    "enbw_sigan": None,  # Defaults to sample rate
    "noise_figure_sigan": 0,
    "1db_compression_sigan": 100,
}

DEFAULT_SENSOR_CALIBRATION = {
    "gain_sensor": None,  # Defaults to sigan gain
    "enbw_sensor": None,  # Defaults to sigan enbw
    "noise_figure_sensor": None,  # Defaults to sigan noise figure
    "1db_compression_sensor": None,  # Defaults to sigan compression + preselector gain
    "gain_preselector": 0,
    "noise_figure_preselector": 0,
    "1db_compression_preselector": 100,
}

WAVEFORMS = {
    "sine": lambda n, tone_offset, rate: np.exp(n * 2j * np.pi * tone_offset / rate),
    "square": lambda n, tone_offset, rate: np.sign(WAVEFORMS["sine"](n, tone_offset, rate)),
    "const": lambda n, tone_offset, rate: 1 + 1j,
    "ramp": lambda n, tone_offset, rate:
            2*(n*(tone_offset/rate) - np.floor(float(0.5 + n*(tone_offset/rate))))
}

class USRPRadio(RadioInterface):
    @property
    def last_calibration_time(self):
        if self.sensor_calibration:
            return utils.convert_string_to_millisecond_iso_format(
                self.sensor_calibration.calibration_datetime
            )
        return None

    @property
    def overload(self):
        return self._sigan_overload or self._sensor_overload

    @property
    def capture_time(self):
        return self._capture_time

    # Define thresholds for determining ADC overload for the sigan
    ADC_FULL_RANGE_THRESHOLD = 0.98  # ADC scale -1<sample<1, magnitude threshold = 0.98
    ADC_OVERLOAD_THRESHOLD = (
        0.01  # Ratio of samples above the ADC full range to trigger overload
    )

    def __init__(
        self,
        sensor_cal_file=settings.SENSOR_CALIBRATION_FILE,
        sigan_cal_file=settings.SIGAN_CALIBRATION_FILE,
    ):
        self.uhd = None
        self.usrp = None
        self._is_available = False

        self.sensor_calibration_data = None
        self.sigan_calibration_data = None
        self.sensor_calibration = None
        self.sigan_calibration = None

        self.lo_freq = None
        self.dsp_freq = None
        self._sigan_overload = False
        self._sensor_overload = False
        self._capture_time = None

        self.connect()
        self.get_calibration(sensor_cal_file, sigan_cal_file)

    def connect(self):
        if self._is_available:
            return True

        if settings.RUNNING_TESTS or settings.MOCK_RADIO:
            logger.warning("Using mock USRP.")
            # random = settings.MOCK_RADIO_RANDOM
            random = False
            self.usrp = MockUsrp(randomize_values=random)
            self._is_available = True
        else:
            try:
                import uhd

                self.uhd = uhd
            except ImportError:
                logger.warning("uhd not available - disabling radio")
                return False

            usrp_args = "type=b200"  # find any b-series device

            try:
                self.usrp = self.uhd.usrp.MultiUSRP(usrp_args)
            except RuntimeError:
                err = "No device found matching search parameters {!r}\n"
                err = err.format(usrp_args)
                raise RuntimeError(err)

            logger.debug("Using the following USRP:")
            logger.debug(self.usrp.get_pp_string())

            try:
                self._is_available = True
                return True
            except Exception as err:
                logger.exception(err)
                return False

    @property
    def is_available(self):
        return self._is_available

    def get_calibration(self, sensor_cal_file, sigan_cal_file):
        # Set the default calibration values
        self.sensor_calibration_data = DEFAULT_SENSOR_CALIBRATION.copy()
        self.sigan_calibration_data = DEFAULT_SIGAN_CALIBRATION.copy()

        # Try and load sensor/sigan calibration data
        if not settings.RUNNING_TESTS and not settings.MOCK_RADIO:
            try:
                self.sensor_calibration = calibration.load_from_json(sensor_cal_file)
            except Exception as err:
                logger.error(
                    "Unable to load sensor calibration data, reverting to none"
                )
                logger.exception(err)
                self.sensor_calibration = None
            try:
                self.sigan_calibration = calibration.load_from_json(sigan_cal_file)
            except Exception as err:
                logger.error("Unable to load sigan calibration data, reverting to none")
                logger.exception(err)
                self.sigan_calibration = None
        else:  # If in testing, create our own test files
            # from scos_usrp import hardware as test_utils

            dummy_calibration = create_dummy_calibration()
            self.sensor_calibration = dummy_calibration
            self.sigan_calibration = dummy_calibration

    @property
    def sample_rate(self):  # -> float:
        return self.usrp.get_rx_rate()

    @sample_rate.setter
    def sample_rate(self, rate):
        """Sets the sample_rate and the clock_rate based on the sample_rate"""
        self.usrp.set_rx_rate(rate)
        fs_MHz = self.sample_rate / 1e6
        logger.debug("set USRP sample rate: {:.2f} MS/s".format(fs_MHz))
        # Set the clock rate based on calibration
        if self.sigan_calibration is not None:
            clock_rate = self.sigan_calibration.get_clock_rate(rate)
        else:
            clock_rate = self.sample_rate
            # Maximize clock rate while keeping it under 40e6
            while clock_rate <= 40e6:
                clock_rate *= 2
            clock_rate /= 2
        self.clock_rate = clock_rate

    @property
    def clock_rate(self):  # -> float:
        return self.usrp.get_master_clock_rate()

    @clock_rate.setter
    def clock_rate(self, rate):
        self.usrp.set_master_clock_rate(rate)
        clk_MHz = self.clock_rate / 1e6
        logger.debug("set USRP clock rate: {:.2f} MHz".format(clk_MHz))

    @property
    def frequency(self):  # -> float:
        return self.usrp.get_rx_freq()

    @frequency.setter
    def frequency(self, freq):
        self.tune_frequency(freq)

    def tune_frequency(self, rf_freq, dsp_freq=0):
        if isinstance(self.usrp, MockUsrp):
            tune_result = self.usrp.set_rx_freq(rf_freq, dsp_freq)
            logger.debug(tune_result)
        else:
            tune_request = self.uhd.types.TuneRequest(rf_freq, dsp_freq)
            tune_result = self.usrp.set_rx_freq(tune_request)
            # FIXME: report actual values when available - see note below
            msg = "rf_freq: {}, dsp_freq: {}"
            logger.debug(msg.format(rf_freq, dsp_freq))

        # FIXME: uhd.types.TuneResult doesn't seem to be implemented
        #        as of uhd 3.13.1.0-rc1
        #        Fake it til they make it
        # self.lo_freq = tune_result.actual_rf_freq
        # self.dsp_freq = tune_result.actual_dsp_freq
        self.lo_freq = rf_freq
        self.dsp_freq = dsp_freq

    @property
    def gain(self):  # -> float:
        return self.usrp.get_rx_gain()

    @gain.setter
    def gain(self, gain):
        if gain not in VALID_GAINS:
            err = "Requested invalid gain {}. ".format(gain)
            err += "Choose one of {!r}.".format(VALID_GAINS)
            logger.error(err)
            return

        self.usrp.set_rx_gain(gain)
        msg = "set USRP gain: {:.1f} dB"
        logger.debug(msg.format(self.usrp.get_rx_gain()))

    def recompute_calibration_data(self):
        """Set the calibration data based on the currently tuning"""

        # Try and get the sensor calibration data
        if self.sensor_calibration is not None:
            self.sensor_calibration_data.update(
                self.sensor_calibration.get_calibration_dict(
                    sample_rate=self.sample_rate,
                    lo_frequency=self.frequency,
                    gain=self.gain,
                )
            )
        else:
            self.sensor_calibration_data = DEFAULT_SENSOR_CALIBRATION.copy()

        # Try and get the sigan calibration data
        if self.sigan_calibration is not None:
            self.sigan_calibration_data.update(
                self.sigan_calibration.get_calibration_dict(
                    sample_rate=self.sample_rate,
                    lo_frequency=self.frequency,
                    gain=self.gain,
                )
            )
        else:
            self.sigan_calibration_data = DEFAULT_SIGAN_CALIBRATION.copy()

        # Catch any defaulting calibration values for the sigan
        if self.sigan_calibration_data["gain_sigan"] is None:
            self.sigan_calibration_data["gain_sigan"] = self.gain
        if self.sigan_calibration_data["enbw_sigan"] is None:
            self.sigan_calibration_data["enbw_sigan"] = self.sample_rate

        # Catch any defaulting calibration values for the sensor
        if self.sensor_calibration_data["gain_sensor"] is None:
            self.sensor_calibration_data["gain_sensor"] = self.sigan_calibration_data[
                "gain_sigan"
            ]
        if self.sensor_calibration_data["enbw_sensor"] is None:
            self.sensor_calibration_data["enbw_sensor"] = self.sigan_calibration_data[
                "enbw_sigan"
            ]
        if self.sensor_calibration_data["noise_figure_sensor"] is None:
            self.sensor_calibration_data[
                "noise_figure_sensor"
            ] = self.sigan_calibration_data["noise_figure_sigan"]
        if self.sensor_calibration_data["1db_compression_sensor"] is None:
            self.sensor_calibration_data["1db_compression_sensor"] = (
                self.sensor_calibration_data["gain_preselector"]
                + self.sigan_calibration_data["1db_compression_sigan"]
            )

    def create_calibration_annotation(self):
        annotation_md = {
            "ntia-core:annotation_type": "CalibrationAnnotation",
            "ntia-sensor:gain_sigan": self.sigan_calibration_data["gain_sigan"],
            "ntia-sensor:noise_figure_sigan": self.sigan_calibration_data[
                "noise_figure_sigan"
            ],
            "ntia-sensor:1db_compression_point_sigan": self.sigan_calibration_data[
                "1db_compression_sigan"
            ],
            "ntia-sensor:enbw_sigan": self.sigan_calibration_data["enbw_sigan"],
            "ntia-sensor:gain_preselector": self.sensor_calibration_data[
                "gain_preselector"
            ],
            "ntia-sensor:noise_figure_sensor": self.sensor_calibration_data[
                "noise_figure_sensor"
            ],
            "ntia-sensor:1db_compression_point_sensor": self.sensor_calibration_data[
                "1db_compression_sensor"
            ],
            "ntia-sensor:enbw_sensor": self.sensor_calibration_data["enbw_sensor"],
        }
        return annotation_md

    def configure(self, action_name):
        pass

    def check_sensor_overload(self, data):
        measured_data = data.astype(np.complex64)
        time_domain_avg_power = 10 * np.log10(np.mean(np.abs(measured_data) ** 2))
        time_domain_avg_power += (
            10 * np.log10(1 / (2 * 50)) + 30
        )  # Convert log(V^2) to dBm
        self._sensor_overload = False
        # explicitly check is not None since 1db compression could be 0
        if self.sensor_calibration_data["1db_compression_sensor"] is not None:
            self._sensor_overload = (
                time_domain_avg_power
                > self.sensor_calibration_data["1db_compression_sensor"]
            )

    ## for PN transmitter
    def create_IQdata(self, seed, sampspersymbol, spacing):
        #Variable initiation
        N = (2**9) - 1
        x1 = np.zeros(N+1)
        x1[0] = seed[0]
        x2 = np.zeros(N+1)
        x2[0] = seed[1]
        x3 = np.zeros(N+1)
        x3[0] = seed[2]
        x4 = np.zeros(N+1)
        x4[0] = seed[3]
        x5 = np.zeros(N+1)
        x5[0] = seed[4]
        x6 = np.zeros(N+1)
        x6[0] = seed[5]
        x7 = np.zeros(N+1)
        x7[0] = seed[6]
        x8 = np.zeros(N+1)
        x8[0] = seed[7]
        x9 = np.zeros(N+1)
        x9[0] = seed[8]
        #Linear shift register
        for i in range(N):
            x1[i + 1] = x2[i]
            x2[i + 1] = x3[i]
            x3[i + 1] = x4[i]
            x4[i + 1] = x5[i]
            x5[i + 1] = x6[i]
            x6[i + 1] = x7[i]
            x7[i + 1] = x8[i]
            x8[i + 1] = x9[i]
            x9[i + 1] = (x1[i] + x5[i]) % 2
        #Binary Phase Shift Key
        #Use x9 for PN sequence
        for i in range(N+1):
            if x9[i] == 0:
                x9[i] = 1
            else:
                x9[i] = -1
        # Increase samples per symbol
        inc = sampspersymbol  #increase scale factor
        data = np.empty(inc * len(x9)-1)
        for i in range(len(x9)-1):
            val = x9[i]
            for j in range(inc):
                data[i * inc + j] = val
        #Add spacing if specified
        if spacing:
            sp = spacing
            data = np.append(data, np.zeros(sp))
        return data


    def acquire_time_domain_samples(
        self, num_samples, num_samples_skip=0, retries=5
    ):  # -> np.ndarray:
        """Aquire num_samples_skip+num_samples samples and return the last num_samples"""
        self._sigan_overload = False
        self._capture_time = None
        # Get the calibration data for the acquisition
        self.recompute_calibration_data()

        # Compute the linear gain
        db_gain = self.sensor_calibration_data["gain_sensor"]
        linear_gain = 10 ** (db_gain / 20.0)

        # Try to acquire the samples
        max_retries = retries
        while True:
            # No need to skip initial samples when simulating the radio
            if settings.RUNNING_TESTS or settings.MOCK_RADIO:
                nsamps = num_samples
            else:
                nsamps = num_samples + num_samples_skip

            self._capture_time = utils.get_datetime_str_now()
            samples = self.usrp.recv_num_samps(
                nsamps,  # number of samples
                self.frequency,  # center frequency in Hz
                self.sample_rate,  # sample rate in samples per second
                [0],  # channel list
                self.gain,  # gain in dB
            )
            # usrp.recv_num_samps returns a numpy array of shape
            # (n_channels, n_samples) and dtype complex64
            assert samples.dtype == np.complex64
            assert len(samples.shape) == 2 and samples.shape[0] == 1
            data = samples[0]  # isolate data for channel 0
            data_len = len(data)

            if not settings.RUNNING_TESTS and not settings.MOCK_RADIO:
                data = data[num_samples_skip:]

            if not len(data) == num_samples:
                if retries > 0:
                    msg = "USRP error: requested {} samples, but got {}."
                    logger.warning(msg.format(num_samples + num_samples_skip, data_len))
                    logger.warning("Retrying {} more times.".format(retries))
                    retries = retries - 1
                else:
                    err = "Failed to acquire correct number of samples "
                    err += "{} times in a row.".format(max_retries)
                    raise RuntimeError(err)
            else:
                logger.debug("Successfully acquired {} samples.".format(num_samples))

                # Check IQ values versus ADC max for sigan compression
                self._sigan_overload = False
                i_samples = np.abs(np.real(data))
                q_samples = np.abs(np.imag(data))
                i_over_threshold = np.sum(i_samples > self.ADC_FULL_RANGE_THRESHOLD)
                q_over_threshold = np.sum(q_samples > self.ADC_FULL_RANGE_THRESHOLD)
                total_over_threshold = i_over_threshold + q_over_threshold
                ratio_over_threshold = float(total_over_threshold) / num_samples
                if ratio_over_threshold > self.ADC_OVERLOAD_THRESHOLD:
                    self._sigan_overload = True

                # Scale the data back to RF power and return it
                data /= linear_gain
                return data

    def acquire_gps_samples(self, n, arg_start_time_goal, retries=5):  # -> np.ndarray:
        """Aquire n samples and return the last n"""
        self._sigan_overload = False
        self._capture_time = None

        # Get the calibration data for the acquisition
        self.recompute_calibration_data()

        # Compute the linear gain
        db_gain = self.sensor_calibration_data["gain_sensor"]
        linear_gain = 10 ** (db_gain / 20.0)

        # Try to acquire the samples
        max_retries = retries
        while True:
            ## no skipping samples with gps sync 
            num_requested_samples = int(n)

            ## ^^^ acquire_samples() also hardcoded as a single channel 0 ^^^
            channel = 0 
            self.usrp.set_clock_source("gpsdo", 0)
            self.usrp.set_time_source("gpsdo", 0)
            self.usrp.set_rx_rate(self.sample_rate, channel)
            self.usrp.set_rx_freq(self.uhd.types.TuneRequest(self.frequency), channel)
            self.usrp.set_rx_gain(self.gain, channel)
            #self.usrp.set_rx_bandwidth(14e6)
            self.usrp.set_rx_antenna("TX/RX", 0)

            ## sleep for a second after setup
            time.sleep(1)

            ## check lock on gps  
            end_time = datetime.now() + timedelta(milliseconds=CLOCK_TIMEOUT)
            ref_locked = self.usrp.get_mboard_sensor("ref_locked", 0).to_bool()
            logger.debug("Waiting for reference lock.")
            while (not ref_locked) and (datetime.now() < end_time):
                time.sleep(1e-3)
                ref_locked = self.usrp.get_mboard_sensor("ref_locked", 0).to_bool()
            if not ref_locked:
                logger.error("No reference lock.")
                raise RuntimeError("Reference lock could not be acquired.")
            logger.debug("Reference locked.")
            logger.debug("Check GPS locked.")
            gps_locked = self.usrp.get_mboard_sensor("gps_locked", 0).to_bool()
            if gps_locked:
                logger.debug("GPS Locked")
            else:
                raise RuntimeError("GPS lock could not be acquired.")
                logger.error("GPS not locked")

            ## set time to gps time
            gps_time = self.uhd.types.TimeSpec(self.usrp.get_mboard_sensor("gps_time", 0).to_int() + 1)
            self.usrp.set_time_next_pps(gps_time)

            ## sleep after setting GPS time
            time.sleep(2)

            samples = np.empty((1, num_requested_samples), dtype=np.complex64)
    
            stream_args = self.uhd.usrp.StreamArgs("fc32", "sc16") ## chose these from the example in https://files.ettus.com/manual/page_converters.html
            stream_args.channels = (0,) ## Note if youre doing mimo this will change e.g., [0,1] 
            md = self.uhd.types.RXMetadata()
            rx_stream = self.usrp.get_rx_stream(stream_args)
            samps_per_buff = rx_stream.get_max_num_samps() ## 2040

            ## set start time
            start_time_goal = self.uhd.types.TimeSpec(arg_start_time_goal)
            ## prime the stream command
            stream_cmd = self.uhd.types.StreamCMD(self.uhd.types.StreamMode.start_cont)
            stream_cmd.stream_now = False
            stream_cmd.time_spec = start_time_goal

            ## wait until within 0.1 secs of start_time_goal
            while True:
                time.sleep(0.01)
                time_check = self.usrp.get_time_last_pps(0) ## 0 is mboard number (i believe)
                if time_check.get_real_secs() >= start_time_goal.get_real_secs() - 1:
                    time.sleep(0.9)
                    break
                #print("current time is    %f" % time_check.get_real_secs())
            
            self._capture_time = datetime.utcnow() 
            ## issue stream command
            rx_stream.issue_stream_cmd(stream_cmd)

            ## start receiving
            num_received_samples = 0
            while num_received_samples < num_requested_samples:
                ## recv() has a default timeout of 0.1 seconds, with no apparent way to change
                num_rx_samps = rx_stream.recv(samples[:,num_received_samples:num_received_samples+samps_per_buff], md)
                if md.error_code != self.uhd.types.RXMetadataErrorCode.none:
                    print(md.strerror())
                    if md.error_code == 1:
                        err = "ERROR_CODE_TIMEOUT occured. Time between issue_stream_cmd() and recv() was longer than 0.1 seconds."
                        raise RuntimeError(err)
                    elif md.error_code == 2:
                        err = "ERROR_CODE_LATE_COMMAND occured. GPS start time already passed."
                        raise RuntimeError(err)
                
                num_received_samples += num_rx_samps

            ## clean up
            stream_cmd = self.uhd.types.StreamCMD(self.uhd.types.StreamMode.stop_cont)
            rx_stream.issue_stream_cmd(stream_cmd)

            recv_buffer = np.zeros((1, samps_per_buff), dtype=np.complex64)
            samps = 1
            while samps:
                samps = rx_stream.recv(recv_buffer, md)
                
            rx_stream = None

            assert samples.dtype == np.complex64
            assert len(samples.shape) == 2 and samples.shape[0] == 1
            data = samples[0]  # isolate data for channel 0
            data_len = len(data)


            if not len(data) == n:
                if retries > 0:
                    msg = "USRP error: requested {} samples, but got {}."
                    logger.warning(msg.format(n, data_len))
                    logger.warning("Retrying {} more times.".format(retries))
                    retries = retries - 1
                else:
                    err = "Failed to acquire correct number of samples "
                    err += "{} times in a row.".format(max_retries)
                    raise RuntimeError(err)
            else:
                logger.debug("Successfully acquired {} samples.".format(n))

                # Check IQ values versus ADC max for sigan compression
                self._sigan_overload = False
                i_samples = np.abs(np.real(data))
                q_samples = np.abs(np.imag(data))
                i_over_threshold = np.sum(i_samples > self.ADC_FULL_RANGE_THRESHOLD)
                q_over_threshold = np.sum(q_samples > self.ADC_FULL_RANGE_THRESHOLD)
                total_over_threshold = i_over_threshold + q_over_threshold
                ratio_over_threshold = float(total_over_threshold) / n
                if ratio_over_threshold > self.ADC_OVERLOAD_THRESHOLD:
                    self._sigan_overload = True

                # Scale the data back to RF power and return it
                data /= linear_gain
                return data


    ### Vadim's PN transmit code below
    def transmit_pn(self, seed, sampspersymbol, spacing, duration_ms):
        # """TX samples based on input arguments"""
        #save IQdata

        data = self.create_IQdata(seed, sampspersymbol, spacing)
        np.reshape(data, (len(data),1))
        # if args.save:
        #     if args.output_file == None:
        #         print("Could not save IQ data, please specify a file name")
        #     else:
        #         with open(args.output_file, 'wb') as out_file:
        #         if args.numpy:
        #                 np.save(out_file, data, allow_pickle=False, fix_imports=False)
        #         else:
        #             data.tofile(out_file)

        # send transmission
        # usrp.send_waveform(data, args.duration, args.freq, args.rate, args.channels, args.gain)

        ## redo for scos version of uhd
        channel = 0 
        self.usrp.set_clock_source("gpsdo", 0)
        self.usrp.set_time_source("gpsdo", 0)
        self.usrp.set_tx_rate(self.sample_rate, channel)
        self.usrp.set_tx_freq(self.uhd.types.TuneRequest(self.frequency), channel)
        self.usrp.set_tx_gain(self.gain, channel)
        self.usrp.set_tx_antenna("TX/RX", 0)
        ## sleep for a quarter second after setup
        time.sleep(0.25)

        ## create the tx_stream
        stream_args = self.uhd.usrp.StreamArgs("fc32", "sc16")
        stream_args.channels = (0,)
        tx_stream = self.usrp.get_tx_stream(stream_args)
        samps_per_buff = tx_stream.get_max_num_samps()

        ## build buffer
        msg = "max_send_buffer_size {} pn_sample_size {}."
        logger.debug(msg.format(samps_per_buff, len(data)))
        big_buff_size = samps_per_buff // len(data) * len(data)
        big_buff = np.empty(big_buff_size)
        for i in range(big_buff_size):
            big_buff[i] = data[i%len(data)]
        num_buffs = (duration_ms / 1000) * self.sample_rate // big_buff_size

        ## set up metadata
        tx_md = self.uhd.types.TXMetadata()
        tx_md.start_of_burst=True ## is true when it is the first packet in a chain
        tx_md.end_of_burst=False ## is true when it's the last packet in a chain
        tx_md.has_time_spec=True 
        gps_time = self.uhd.types.TimeSpec(self.usrp.get_mboard_sensor("gps_time", 0).to_int() + 1)
        tx_md.time_spec=gps_time ## when to send the first sample

        ## transmit
        for i in range(num_buffs-1):
            samps_sent = tx_stream.send(big_buff, tx_md)
            tx_md.start_of_burst=False
        tx_md.end_of_burst=True
        samps_sent = tx_stream.send(big_buff, tx_md)
        return data