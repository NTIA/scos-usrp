from scos_usrp.hardware.gps_iface import USRPLocation
from scos_usrp.hardware.usrp_sigan import USRPSignalAnalyzer

sigan = USRPSignalAnalyzer()
gps = USRPLocation(sigan)
