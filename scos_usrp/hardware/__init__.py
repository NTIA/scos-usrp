from scos_actions.actions.interfaces.signals import register_component_with_status

from scos_usrp.hardware.gps_iface import USRPLocation
from scos_usrp.hardware.usrp_sigan import USRPSignalAnalyzer

sigan = USRPSignalAnalyzer()
register_component_with_status.send(sigan, component=sigan)
gps = USRPLocation(sigan)
