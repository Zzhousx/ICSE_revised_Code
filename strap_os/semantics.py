"""Data and dependency adapters; all semantic functions execute from the pinned clone."""
from pathlib import Path
import sys,types,importlib,json,math,xml.etree.ElementTree as ET
from types import SimpleNamespace as NS
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';CLONE=OUT/'strap'
sys.dont_write_bytecode=True
sys.path[:0]=[str(OUT/'generated'),str(CLONE)]

def nested(x):
 if isinstance(x,dict):return NS(**{k:nested(v) for k,v in x.items()})
 if isinstance(x,list):return [nested(v) for v in x]
 return x
class Message:
 def ParseFromString(self,raw):self.__dict__.update(nested(json.loads(raw.decode())).__dict__)
class Reader:
 records={}
 def __init__(self,path):self.items=self.records[path]
 def read_messages(self):return iter(self.items)
def ensure(name):
 if name in sys.modules:return sys.modules[name]
 try:return importlib.import_module(name)
 except ModuleNotFoundError:
  m=types.ModuleType(name);m.__path__=[];sys.modules[name]=m
  if '.' in name:
   parent,child=name.rsplit('.',1);setattr(ensure(parent),child,m)
  return m
def install_input_adapters():
 from modules.map.proto import map_pb2
 for path,symbols in {
 'modules.perception.proto.perception_obstacle_pb2':['PerceptionObstacles'],
 'modules.perception.proto.traffic_light_detection_pb2':['TrafficLightDetection'],
 'modules.prediction.proto.prediction_obstacle_pb2':['PredictionObstacles'],
 'modules.localization.proto.localization_pb2':['LocalizationEstimate'],
 'modules.drivers.proto.sensor_image_pb2':['Image','CompressedImage'],
 'modules.routing.proto.routing_pb2':['RoutingRequest','RoutingResponse'],
 'modules.transform.proto.transform_pb2':['TransformStampeds'],
 'modules.planning.proto.planning_pb2':['ADCTrajectory']}.items():
  m=ensure(path)
  for name in symbols:setattr(m,name,Message)
 ensure('cyber_py').record=NS(RecordReader=Reader)
 def load_map(path,msg):msg.ParseFromString(Path(path).read_bytes());return msg
 ensure('common.proto_utils').get_pb_from_file=load_map
install_input_adapters()
import utils as official_utils
import map_parsing as official_map
import segment_split as official_segments
import prioritization as official_priority

# Published SanFrancisco.unity MapOrigin: position=(255.0007,10.14,39.89137), yaw=-90deg.
ORIGIN_E=552873.9836079116;ORIGIN_N=4182486.0198973147
def apollo_position(x,y,z):return dict(x=ORIGIN_E+(z-39.89137),y=ORIGIN_N-(x-255.0007),z=y-10.14)
def apollo_heading(unity_yaw):return math.atan2(-math.sin(math.radians(unity_yaw)),math.cos(math.radians(unity_yaw)))
def actor_class(typ,model):
 s=(typ+' '+model).lower()
 if 'pedestrian' in s or 'person' in s:return 3,10
 if 'bicycle' in s or 'cyclist' in s:return 4,7
 if 'motorcycle' in s:return 4,8
 if 'npc' in s or 'vehicle' in s:
  return 5,6 if 'bus' in s else 5 if 'truck' in s else 4 if 'van' in s else 3 if any(x in s for x in ['sedan','jeep','suv','hatchback','car']) else 0
 return 2,2
def parse(text):
 root=ET.fromstring(text);dt=float(root.get('timestep','.5'))
 objs={e.get('name'):e for e in root.findall('.//Entities/ScenarioObject')}
 acts={e.get('name','').replace('Act_','',1):e for e in root.findall('.//ObjectAction')}
 inits={e.get('objectRef'):e for e in root.findall('.//ObjectInitialization')}
 ego=next(n for n,e in objs.items() if e.find('ObjectParameter').get('ObjectType','').lower()=='ego')
 ego_wp=acts[ego].findall('WayPoint');T=len(ego_wp);actors=[]
 def state(name,i):
  wp=acts[name].findall('WayPoint')[i] if name in acts and acts[name].findall('WayPoint') else inits[name]
  p=wp.find('.//ObjectPosition');r=wp.find('.//ObjectRotation');vel=wp.find('.//Velocity')
  if p is None:p=wp.find('.//ObjectInitialPosition')
  if r is None:r=wp.find('.//ObjectInitialRotation')
  xyz=[float(p.get('position'+a)) for a in 'XYZ'];yaw=float(r.get('rotationY'))
  # Velocity is carried when present; official obstacle encoder does not read it.
  v=[float(vel.get('velocity'+a,'0')) for a in 'XYZ'] if vel is not None else [0,0,0]
  return xyz,yaw,v
 for name,obj in objs.items():
  if name==ego:continue
  p=obj.find('ObjectParameter');typ=p.get('ObjectType','') if p is not None else '';model=obj.get('model','');typeid,subtype=actor_class(typ,model)
  actors.append((name,typ,model,typeid,subtype))
 loc=[];obs=[];ego_positions=[]
 for i in range(T):
  xyz,yaw,v=state(ego,i);ego_positions.append(xyz)
  loc.append({'header':{'timestamp_sec':1000000000+i*dt},'pose':{'position':apollo_position(*xyz),'heading':apollo_heading(yaw)}})
  items=[]
  for j,(name,typ,model,typeid,subtype) in enumerate(actors):
   xyz2,yaw2,v2=state(name,i)
   items.append({'id':j+1,'type':typeid,'sub_type':subtype,'position':apollo_position(*xyz2),'theta':apollo_heading(yaw2),'velocity':{'x':v2[2],'y':-v2[0],'z':v2[1]}})
  obs.append({'perception_obstacle':items})
 return dict(loc=loc,obs=obs,dt=dt,actors=actors,ego_positions=ego_positions,root=root)
def messages(parsed,R=None):
 loc=parsed['loc'];obs=parsed['obs']
 if R is not None:
  obs=[{'perception_obstacle':[o for o in om['perception_obstacle'] if math.hypot(o['position']['x']-lm['pose']['position']['x'],o['position']['y']-lm['pose']['position']['y'])<=R]} for lm,om in zip(loc,obs)]
 # Unavailable ancillary channels are empty default messages, used only to permit
 # the official alignment bookkeeping. No scene/planning semantics are encoded.
 rows=[]
 for lm,om in zip(loc,obs):
  for channel,obj in [('/apollo/localization/pose',lm),('/apollo/perception/traffic_light',{'contain_lights':False,'traffic_light':[]}),('/apollo/perception/obstacles',om),('/apollo/planning',{})]:
   rows.append((channel,json.dumps(obj).encode(),'',int(lm['header']['timestamp_sec']*1e9)))
 Reader.records['adapter']=rows
 aligned=official_utils.get_alignment_messages('adapter',test_system='obstacle');Reader.records.clear()
 assert len(aligned[0])==len(loc)
 return aligned[0],aligned[2]
def encode(parsed,map_obj,R=None):
 lm,om=messages(parsed,R);return official_segments.create_actor_vecs(lm,om,map_obj)
def encode_proxy(parsed):
 from shapely.geometry import Polygon
 lm,om=messages(parsed);vectors=[]
 for l,o in zip(lm,om):
  x=l.pose.position.x;y=l.pose.position.y;h=l.pose.heading;f=np.array([math.cos(h),math.sin(h)]);side=np.array([-math.sin(h),math.cos(h)]);p=np.array([x,y])
  def rect(a,b):return Polygon([p+f*t+side*s for t,s in [(a,-1.75),(b,-1.75),(b,1.75),(a,1.75)]])
  proxy=NS(lanes={'proxy_ego_lane':[rect(-50,50),0]},crosswalk={'proxy_near_forward_patch':([],rect(0,5))})
  vectors+=official_segments.create_actor_vecs([l],[o],proxy)
 return vectors
def segmented(raw,dt,window=3):
 smooth=official_utils.vec_denoise_v3(raw,window)
 loc=[nested({'header':{'timestamp_sec':1000000000+i*dt}}) for i in range(len(raw))]
 # The original call receives original localization/time arrays, even though
 # the smoothing function shortens the actor vector sequence.
 data={'actor_vec':smooth,'time':[i*dt for i in range(len(raw))],'localization_msgs':loc}
 result=official_segments.record2segments(data,'obstacle',kept_length=3)
 return smooth,result
