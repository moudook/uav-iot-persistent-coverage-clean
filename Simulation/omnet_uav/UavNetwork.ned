package inet.examples.uav_omnet;



import inet.networklayer.configurator.ipv4.Ipv4NetworkConfigurator;
import inet.node.inet.AdhocHost;
import inet.node.inet.StandardHost;
import inet.physicallayer.wireless.ieee80211.packetlevel.Ieee80211ScalarRadioMedium;
import inet.mobility.single.BonnMotionMobility;

network UavNetwork
{
    parameters:
        int numUavs;
        int numSensors;
        @display("bgb=800,800");
    submodules:
        configurator: Ipv4NetworkConfigurator {
            parameters:
                @display("p=50,50;is=s");
        }
        radioMedium: Ieee80211ScalarRadioMedium {
            parameters:
                @display("p=50,150;is=s");
        }
        uav[numUavs]: AdhocHost {
            parameters:
                @display("r=,,#0080ff;p=400,400");
                mobility.typename = "BonnMotionMobility";
                mobility.traceFile = "traces/episode_000.movements";
                mobility.nodeId = index;
        }
        sensor[numSensors]: AdhocHost {
            parameters:
                @display("r=,,#808080;p=400,400");
        }
        sink: StandardHost {
            parameters:
                @display("r=,,#ff0000;p=400,400");
        }
}
