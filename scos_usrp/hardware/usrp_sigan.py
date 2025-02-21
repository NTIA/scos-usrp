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
import subprocess
from typing import Dict, Optional

import numpy as np
from its_preselector.web_relay import WebRelay
from scos_actions import utils
from scos_actions.hardware.sigan_iface import SignalAnalyzerInterface

from scos_usrp import __package__ as SCOS_USRP_NAME
from scos_usrp import __version__ as SCOS_USRP_VERSION
from scos_usrp import settings
from scos_usrp.hardware.mocks.usrp_block import MockUsrp

logger = logging.getLogger(__name__)
logger.debug(f"USRP_CONNECTION_ARGS = {settings.USRP_CONNECTION_ARGS}")

# Testing determined these gain values provide a good mix of sensitivity and
# dynamic range performance
VALID_GAINS = (0, 20, 40, 60)


class USRPSignalAnalyzer(SignalAnalyzerInterface):
    # Define thresholds for determining ADC overload for the sigan
    ADC_FULL_RANGE_THRESHOLD = 0.98  # ADC scale -1<sample<1, magnitude threshold = 0.98
    ADC_OVERLOAD_THRESHOLD = (
        0.01  # Ratio of samples above the ADC full range to trigger overload
    )

    def __init__(
        self,
        switches: Optional[Dict[str, WebRelay]] = None,
    ):
        super().__init__(switches)
        self._plugin_version = SCOS_USRP_VERSION
        self._plugin_name = SCOS_USRP_NAME
        self._model = "Unknown"
        self._api_version = "Unknown"
        self._firmware_version = "Unknown"
        self.uhd = None
        self.usrp = None
        self._is_available = False
        self.lo_freq = None
        self.dsp_freq = None
        self._capture_time = None
        self.requested_sample_rate = 0
        self.requested_frequency = 0
        self.requested_gain = 0
        self.requested_clock_rate = 0
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
            self._model = self.usrp.get_mboard_name()
            try:
                self._is_available = True
                return True
            except Exception as err:
                logger.exception(err)
                return False
            
    @property
    def firmware_version(self) -> str:
        """Returns the version of the signal analyzer firmware."""
        # firmware version
        # based on https://github.com/EttusResearch/uhd/blob/master/host/utils/uhd_usrp_probe.cpp
        tree = self.usrp.get_tree()
        addr = tree.list("/mboards")[0]
        path = f"/mboards/{addr}"
        return tree.access_str(path + "/fw_version").get()

    @property
    def api_version(self) -> str:
        """Returns the version of the underlying signal analyzer API."""
        raw_version = subprocess.run(["dpkg", "-s", "python3-uhd"], capture_output=True, text=True).stdout
        new_line_split = raw_version.split("\n")
        for line in new_line_split:
            if line.startswith("Version"):
                return line.split(":")[1].strip()

    @property
    def plugin_version(self):
        """Returns the current version of scos-usrp."""
        return self._plugin_version

    @property
    def plugin_name(self) -> str:
        """Returns the current package name of scos-usrp."""
        return self._plugin_name

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
        clock_rate = rate
        # Maximize clock rate while keeping it under 40e6
        while clock_rate <= 40e6:
            clock_rate *= 2
        clock_rate /= 2
        self.clock_rate = clock_rate
        if round(self.clock_rate, 1) != round(clock_rate, 1):
            raise Exception(
                f"Clock rate {self.clock_rate} does not match requested rate {clock_rate}!"
            )
        self.requested_sample_rate = rate
        self.usrp.set_rx_rate(rate)
        fs_MSps = self.sample_rate / 1e6
        logger.debug("set USRP sample rate: {:.2f} MSps".format(fs_MSps))
        if round(self.sample_rate, 1) != round(self.requested_sample_rate, 1):
            raise Exception(
                f"Sample rate {self.sample_rate} does not match requested rate {self.requested_sample_rate}!"
            )

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
            msg = "Requested invalid gain {}. ".format(gain)
            msg += "It is recommended to choose one of {!r}.".format(VALID_GAINS)
            logger.warning(msg)
        self.requested_gain = gain
        self.usrp.set_rx_gain(gain)
        msg = "set USRP gain: {:.1f} dB"
        logger.debug(msg.format(self.usrp.get_rx_gain()))

    def acquire_time_domain_samples(self, num_samples: int, num_samples_skip: int = 0):
        """Acquire num_samples_skip+num_samples samples and return the last num_samples

        :type num_samples: int
        :param num_samples: Number of samples to acquire

        :type num_samples_skip: int
        :param num_samples_skip: Skip samples to allow signal analyzer DC offset and IQ imbalance algorithms to take effect

        :rtype: dictionary containing the following:
            data - (list) measurement data
            overload - (boolean) True if overload occurred, otherwise False
            frequency - (float) Measurement center frequency in hertz
            gain - (float) Measurement signal analyzer gain setting in dB
            sample_rate - (float) Measurement sample rate in samples per second
            capture_time - (string) Measurement capture time
            calibration_annotation - (dict) SigMF calibration annotation
        """
        sigan_overload = False
        self._capture_time = None
        # Get the calibration data for the acquisition
        logger.debug(
            "Using requested sample rate of " + str(self.requested_sample_rate)
        )
        nsamps = int(num_samples)
        nskip = int(num_samples_skip)

        # Try to acquire the samples
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
                msg = f"USRP error: requested {num_samples + num_samples_skip} samples, but got {data_len}."
                logger.error(msg)
                raise RuntimeError(msg)
            else:
                logger.debug("Successfully acquired {} samples.".format(num_samples))

                # Check IQ values versus ADC max for sigan compression
                sigan_overload = False
                i_samples = np.abs(np.real(data))
                q_samples = np.abs(np.imag(data))
                i_over_threshold = np.sum(i_samples > self.ADC_FULL_RANGE_THRESHOLD)
                q_over_threshold = np.sum(q_samples > self.ADC_FULL_RANGE_THRESHOLD)
                total_over_threshold = i_over_threshold + q_over_threshold
                ratio_over_threshold = float(total_over_threshold) / num_samples
                if ratio_over_threshold > self.ADC_OVERLOAD_THRESHOLD:
                    sigan_overload = True
                    logger.warning("Signal Analyzer overload occurred!")

                measurement_result = {
                    "data": data,
                    "overload": sigan_overload,
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
