"""Test aspects of SignalAnalyzerInterface with mocked USRP."""

import pytest

from scos_usrp.hardware.usrp_sigan import USRPSignalAnalyzer


class TestUSRP:
    # Ensure we write the test cal file and use mocks
    setup_complete = False

    @pytest.fixture(autouse=True)
    def setup_mock_usrp(self):
        """Create the mock USRP"""
        self.rx = USRPSignalAnalyzer()

        # Only setup once
        if self.setup_complete:
            return

        # Create the SignalAnalyzerInterface with the mock usrp_block and get the sigan
        # usrp_iface.connect()
        if not self.rx.is_available:
            raise RuntimeError("Receiver is not available.")

        # Alert that the setup was complete
        self.setup_complete = True

    def test_tune_result(self):
        """Check that the tuning is correct"""
        # Check that the setup was completed
        assert self.setup_complete, "Setup was not completed"

        # Use a positive DSP frequency
        f_lo = 1.0e9
        f_dsp = 1.0e6
        self.rx.tune_frequency(f_lo, f_dsp)
        assert f_lo == self.rx.lo_freq and f_dsp == self.rx.dsp_freq

        # Use a 0Hz for DSP frequency
        f_lo = 1.0e9
        f_dsp = 0.0
        self.rx.frequency = f_lo
        assert f_lo == self.rx.lo_freq and f_dsp == self.rx.dsp_freq

        # Use a negative DSP frequency
        f_lo = 1.0e9
        f_dsp = -1.0e6
        self.rx.tune_frequency(f_lo, f_dsp)
        assert f_lo == self.rx.lo_freq and f_dsp == self.rx.dsp_freq

    def test_set_sample_rate_also_sets_clock_rate(self):
        """Setting sample_rate should adjust clock_rate"""

        # Check that the setup was completed
        assert self.setup_complete, "Setup was not completed"

        expected_clock_rate = 30720000

        # Set the sample rate and check the clock rate
        self.rx.sample_rate = 15360000
        observed_clock_rate = self.rx.clock_rate

        assert expected_clock_rate == observed_clock_rate
