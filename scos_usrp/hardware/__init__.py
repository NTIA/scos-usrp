from scos_usrp.hardware.gps_iface import USRPLocation
from scos_usrp.hardware.usrp_radio import USRPRadio

radio = USRPRadio()
gps = USRPLocation(radio)
