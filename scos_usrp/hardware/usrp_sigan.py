"""Maintains a persistent connection to the USRP.

Example usage:
    >>> from scos_usrp.hardware import sigan
    >>> sigan.is_available
    True
    >>> rx = sigan
    >>> rx.sample_rate = 10e6
    >>> rx.frequency = 700e6
    >>> rx.gain = 40
    >>> samples = rx.acquire_time_domain_samples(1000)
"""

import logging
from typing import Dict, Optional

import numpy as np
from its_preselector.web_relay import WebRelay
from scos_actions import utils
from scos_actions.calibration.calibration import Calibration
from scos_actions.hardware.sigan_iface import SignalAnalyzerInterface

from scos_usrp import __version__ as SCOS_USRP_VERSION
from scos_usrp import settings
from scos_usrp.hardware.mocks.usrp_block import MockUsrp

logger = logging.getLogger(__name__)
logger.debug(f"USRP_CONNECTION_ARGS = {settings.USRP_CONNECTION_ARGS}")

# Testing determined these gain values provide a good mix of sensitivity and
# dynamic range performance
VALID_GAINS = (0, 20, 40, 60)


class USRPSignalAnalyzer(SignalAnalyzerInterface):

    @property
    def overload(self):
        """Returns True if overload occurred, otherwise returns False."""
        return self._sigan_overload or self._sensor_overload

    # Define thresholds for determining ADC overload for the sigan
    ADC_FULL_RANGE_THRESHOLD = 0.98  # ADC scale -1<sample<1, magnitude threshold = 0.98
    ADC_OVERLOAD_THRESHOLD = (
        0.01  # Ratio of samples above the ADC full range to trigger overload
    )

    def __init__(
        self,
        sensor_cal: Calibration = None,
        sigan_cal: Calibration = None,
        switches: Optional[Dict[str, WebRelay]] = None,
    ):
        super().__init__(sensor_cal, sigan_cal, switches)
        self._plugin_version = SCOS_USRP_VERSION
        self.uhd = None
        self.usrp = None
        self._is_available = False
        self.lo_freq = None
        self.dsp_freq = None
        self._sigan_overload = False
        self._sensor_overload = False
        self._capture_time = None
        self.requested_sample_rate = 0
        self.requested_frequency = 0
        self.requested_gain = 0
        self.requested_clock_rate = 0
        self.sensor_calibration_data = None
        self.sigan_calibration_data = None
        self.connect()

    def connect(self):
        if self._is_available:
            return True

        if settings.RUNNING_TESTS or settings.MOCK_SIGAN:
            logger.warning("Using mock USRP.")
            random = settings.MOCK_SIGAN_RANDOM
            self.usrp = MockUsrp(randomize_values=random)
            self._is_available = True
        else:
            try:
                import uhd

                self.uhd = uhd
            except ImportError:
                logger.warning("uhd not available - disabling signal analyzer")
                return False

            usrp_args = (
                f"type=b200,{settings.USRP_CONNECTION_ARGS}"  # find any b-series device
            )
            logger.debug(f"usrp_args = {usrp_args}")

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
    def plugin_version(self):
        """Returns the current version of scos-usrp."""
        return self._plugin_version

    @property
    def is_available(self):
        """Returns True if initialized and ready to make measurements, otherwise returns False."""
        return self._is_available

    @property
    def sample_rate(self):
        """Returns the currently configured sample rate in samples per second."""
        return self.usrp.get_rx_rate()

    @sample_rate.setter
    def sample_rate(self, rate):
        """Sets the sample_rate and the clock_rate based on the sample_rate

        :type sample_rate: float
        :param sample_rate: Sample rate in samples per second
        """
        self.requested_sample_rate = rate
        self.usrp.set_rx_rate(rate)
        fs_MSps = self.sample_rate / 1e6
        logger.debug("set USRP sample rate: {:.2f} MSps".format(fs_MSps))
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
    def clock_rate(self):
        """Returns the currently configured clock rate in hertz."""
        return self.usrp.get_master_clock_rate()

    @clock_rate.setter
    def clock_rate(self, rate):
        """Sets the signal analyzer clock rate.

        :type rate: float
        :param rate: Clock rate in hertz
        """
        self.requested_clock_rate = rate
        self.usrp.set_master_clock_rate(rate)
        clk_MHz = self.clock_rate / 1e6
        logger.debug("set USRP clock rate: {:.2f} MHz".format(clk_MHz))

    @property
    def frequency(self):
        """Returns the currently configured center frequency in hertz."""
        return self.usrp.get_rx_freq()

    @frequency.setter
    def frequency(self, freq):
        """Sets the signal analyzer frequency.

        :type freq: float
        :param freq: Frequency in hertz
        """
        self.requested_frequency = freq
        self.tune_frequency(freq)

    def tune_frequency(self, rf_freq, dsp_freq=0):
        """Tunes the signal analyzer as close as possible to the desired frequency.

        :type rf_freq: float
        :param rf_freq: Desired frequency in hertz

        :type dsp_freq: float
        :param dsp_freq: LO offset frequency in hertz
        """
        if isinstance(self.usrp, MockUsrp):
            tune_result = self.usrp.set_rx_freq(rf_freq, dsp_freq)
            logger.debug(tune_result)
        else:
            tune_request = self.uhd.types.TuneRequest(rf_freq, dsp_freq)
            tune_result = self.usrp.set_rx_freq(tune_request)
            msg = "rf_freq: {}, dsp_freq: {}"
            logger.debug(msg.format(rf_freq, dsp_freq))

        self.lo_freq = rf_freq
        self.dsp_freq = dsp_freq

    @property
    def gain(self):
        """Returns the currently configured gain setting in dB."""
        return self.usrp.get_rx_gain()

    @gain.setter
    def gain(self, gain):
        """Sets the signal analyzer gain setting.

        :type gain: float
        :param gain: Gain in dB
        """
        if gain not in VALID_GAINS:
            err = "Requested invalid gain {}. ".format(gain)
            err += "Choose one of {!r}.".format(VALID_GAINS)
            logger.error(err)
            return
        self.requested_gain = gain
        self.usrp.set_rx_gain(gain)
        msg = "set USRP gain: {:.1f} dB"
        logger.debug(msg.format(self.usrp.get_rx_gain()))

    def check_sensor_overload(self, data, cal_adjust=True):
        """Check for sensor overload in the measurement data."""
        measured_data = data.astype(np.complex64)

        time_domain_avg_power = 10 * np.log10(np.mean(np.abs(measured_data) ** 2))
        time_domain_avg_power += (
            10 * np.log10(1 / (2 * 50)) + 30
        )  # Convert log(V^2) to dBm
        self._sensor_overload = False
        # explicitly check is not None since 1db compression could be 0
        if (
            cal_adjust
            and "1db_compression_point" in self.sensor_calibration_data
            and self.sensor_calibration_data["1db_compression_point"] is not None
        ):
            self._sensor_overload = bool(
                time_domain_avg_power
                > self.sensor_calibration_data["1db_compression_point"]
            )

    def acquire_time_domain_samples(
        self, num_samples, num_samples_skip=0, retries=5, cal_adjust: bool = True
    ):
        """Acquire num_samples_skip+num_samples samples and return the last num_samples

        :type num_samples: int
        :param num_samples: Number of samples to acquire

        :type num_samples_skip: int
        :param num_samples_skip: Skip samples to allow signal analyzer DC offset and IQ imbalance algorithms to take effect

        :type retries: int
        :param retries: The number of retries to attempt when failing to acquire samples

        :rtype: dictionary containing the following:
            data - (list) measurement data
            overload - (boolean) True if overload occurred, otherwise False
            frequency - (float) Measurement center frequency in hertz
            gain - (float) Measurement signal analyzer gain setting in dB
            sample_rate - (float) Measurement sample rate in samples per second
            capture_time - (string) Measurement capture time
            calibration_annotation - (dict) SigMF calibration annotation
        """
        self._sigan_overload = False
        self._capture_time = None
        # Get the calibration data for the acquisition
        logger.debug(
            "Using requested sample rate of " + str(self.requested_sample_rate)
        )
        cal_params = []

        if cal_adjust:
            if not (settings.RUNNING_TESTS or settings.MOCK_SIGAN):
                cal_params = self.sensor_calibration.calibration_parameters
            try:
                logger.debug(f"Using cal params: {cal_params}")
                cal_args = []
                if cal_params is not None:
                    for p in cal_params:
                        cal_args.append(getattr(self, "requested_" + p))
                else:
                    cal_args = None
            except KeyError:
                raise Exception(
                    "One or more required cal parameters is not a valid sigan setting."
                )
            logger.debug(f"Calibration arguments:{cal_args}")
            self.recompute_sensor_calibration_data(cal_args)
            # Compute the linear gain
            db_gain = self.sensor_calibration_data["gain"]
            linear_gain = 10 ** (db_gain / 20.0)
        else:
            linear_gain = 1
        nsamps = int(num_samples)
        nskip = int(num_samples_skip)

        # Try to acquire the samples
        max_retries = retries
        while True:
            # No need to skip initial samples when simulating the signal analyzer
            if not settings.RUNNING_TESTS and not settings.MOCK_SIGAN:
                nsamps += nskip

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

            if not settings.RUNNING_TESTS and not settings.MOCK_SIGAN:
                data = data[nskip:]

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
                self.check_sensor_overload(data, cal_adjust)
                measurement_result = {
                    "data": data,
                    "overload": self.overload,
                    "frequency": self.frequency,
                    "gain": self.gain,
                    "sample_rate": self.sample_rate,
                    "capture_time": self._capture_time,
                }
                return measurement_result

    def healthy(self):
        logger.debug("Performing USRP health check")

        if not self.is_available:
            return False

        try:
            radio_config = self.usrp.get_pp_string()
            logger.debug("Radio config: " + radio_config)
        except Exception as e:
            logger.error("Unable to obtain radio configuration")
            logger.error(e)
            return False

        return True
