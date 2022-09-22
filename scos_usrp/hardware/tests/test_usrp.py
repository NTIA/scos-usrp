"""Test aspects of SignalAnalyzerInterface with mocked USRP."""

import pytest

from scos_usrp.hardware import sigan


class TestUSRP:
    # Ensure we write the test cal file and use mocks
    setup_complete = False

    @pytest.fixture(autouse=True)
    def setup_mock_usrp(self):
        """Create the mock USRP"""

        # Only setup once
        if self.setup_complete:
            return

        # Create the SignalAnalyzerInterface with the mock usrp_block and get the sigan
        # usrp_iface.connect()
        if not sigan.is_available:
            raise RuntimeError("Receiver is not available.")
        self.rx = sigan

        # Alert that the setup was complete
        self.setup_complete = True

    # Ensure the usrp can recover from acquisition errors
    def test_acquire_samples_with_retries(self):
        """Acquire samples should retry without error up to `max_retries`."""

        # Check that the setup was completed
        assert self.setup_complete, "Setup was not completed"

        max_retries = 5
        times_to_fail = 3
        self.rx.sample_rate = 10000000.0
        self.rx.frequency = 650000000.0
        self.rx.gain = 40.0
        self.rx.usrp.set_times_to_fail_recv(times_to_fail)

        try:
            self.rx.acquire_time_domain_samples(1000, retries=max_retries)
        except RuntimeError:
            msg = "Acquisition failing {} times sequentially with {}\n"
            msg += "retries requested should NOT have raised an error."
            msg = msg.format(times_to_fail, max_retries)
            pytest.fail(msg)

        self.rx.usrp.set_times_to_fail_recv(0)

    def test_acquire_samples_fails_when_over_max_retries(self):
        """After `max_retries`, an error should be thrown."""

        # Check that the setup was completed
        assert self.setup_complete, "Setup was not completed"

        max_retries = 5
        times_to_fail = 7
        self.rx.usrp.set_times_to_fail_recv(times_to_fail)
        self.rx.sample_rate = 10000000.0
        self.rx.frequency = 650000000.0
        self.rx.gain = 40.0
        msg = "Acquisition failing {} times sequentially with {}\n"
        msg += "retries requested SHOULD have raised an error."
        msg = msg.format(times_to_fail, max_retries)
        with pytest.raises(RuntimeError):
            self.rx.acquire_time_domain_samples(1000, 1000, max_retries)
            pytest.fail(msg)

        self.rx.usrp.set_times_to_fail_recv(0)

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
