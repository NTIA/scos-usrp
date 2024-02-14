import logging
from unittest.mock import MagicMock, patch

from pytest import approx
from scos_usrp.hardware.gps_iface import USRPLocation

from scos_usrp.hardware.usrp_sigan import USRPSignalAnalyzer

class TestGPS:
    @patch("uhd.usrp.MultiUSRP")
    def test_get_lat_long_returns_location(self, mock_usrp, caplog):
        caplog.set_level(logging.DEBUG)
        get_mboard_sensor_mock = MagicMock()

        def side_effect(value):
            if value == "gps_gpgga":
                gps_gpgga_mock = MagicMock()
                gps_gpgga_mock.value = "$GPGGA,164747.933,3959.707,N,10515.695,W,1,12,1.0,10.0,M,0.0,M,,*7E"
                return gps_gpgga_mock
            elif value == "gps_time":
                return MagicMock(return_value=1587746867)
            return get_mboard_sensor_mock.return_value

        get_mboard_sensor_mock.side_effect = side_effect
        mock_usrp.get_mboard_sensor = get_mboard_sensor_mock
        mock_usrp.get_time_sources = MagicMock(return_value=["gpsdo"])
        mock_usrp.get_time_source = MagicMock(return_value="gpsdo")
        mock_usrp.get_clock_sources = MagicMock(return_value=["gpsdo"])
        mock_usrp.get_clock_source = MagicMock(return_value="gpsdo")
        sigan = USRPSignalAnalyzer()
        sigan.usrp = mock_usrp
        sigan.uhd = MagicMock()
        sigan.uhd.types = MagicMock()
        sigan.uhd.types.TimeSpec = MagicMock()
        gps = USRPLocation(sigan)
        latitude, longitude, height = gps.get_location()
        assert latitude == approx(39.99511463)
        assert longitude == approx(-105.26158690)
        assert height == approx(10.0)

    @patch("uhd.usrp.MultiUSRP")
    def test_get_lat_long_no_gps(self, mock_usrp, caplog):
        caplog.set_level(logging.DEBUG)
        get_mboard_sensor_mock = MagicMock()

        def side_effect(value):
            if value == "gps_gpgga":
                gps_gpgga_mock = MagicMock()
                gps_gpgga_mock.value = None
                return gps_gpgga_mock
            elif value == "gps_time":
                return None
            elif value == "gps_locked":
                return MagicMock(return_value=False)
            return get_mboard_sensor_mock.return_value

        get_mboard_sensor_mock.side_effect = side_effect
        mock_usrp.get_mboard_sensor = get_mboard_sensor_mock
        mock_usrp.get_time_sources = MagicMock(return_value=[""])
        mock_usrp.get_time_source = MagicMock(return_value="")
        mock_usrp.get_clock_sources = MagicMock(return_value=[""])
        mock_usrp.get_clock_source = MagicMock(return_value="")
        sigan = USRPSignalAnalyzer()
        sigan.usrp = mock_usrp
        sigan.uhd = MagicMock()
        sigan.uhd.types = MagicMock()
        sigan.uhd.types.TimeSpec = MagicMock()
        gps = USRPLocation(sigan)
        ret = gps.get_location()
        assert ret == None
