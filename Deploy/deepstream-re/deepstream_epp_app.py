#!/usr/bin/env python3

# Este Script ya lee video de un rtsp o un video file:///
# python3 deepstream_epp_app.py rtsp://admin:TakColombia2020@192.168.1.133/MPEG-4/ch1/main/av_stream

import sys
sys.path.append('../')
import gi
import configparser
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst
#from gi.repository import GLib
from ctypes import *
import time
import sys
import numpy as np
import math
import platform
from common.is_aarch_64 import is_aarch64
from common.bus_call import bus_call
from common.FPS import GETFPS
#from common.utils import long_to_int
import jetson.utils
from threading import Thread
from queue import Queue
import os

import cv2
import pyds
import boto3
import pymysql
import logging
from botocore.client import Config
from datetime import datetime

fps_streams={}
start_time = time.time()

MUXER_OUTPUT_WIDTH=1280#Original:1280
MUXER_OUTPUT_HEIGHT=720 #Original:720
MUXER_BATCH_TIMEOUT_USEC=4000000
TILED_OUTPUT_WIDTH=1280 #Original:1280
TILED_OUTPUT_HEIGHT=720 #Original:720
GST_CAPS_FEATURES_NVMM="memory:NVMM"
OSD_PROCESS_MODE= 0
OSD_DISPLAY_TEXT= 1

PGIE_CLASS_ID_PERSON = 2 # Esta me importa
SGIE_ALERT_ID_PIES = 0
SGIE_ALERT_ID_CABEZA = 1
SGIE_ALERT_ID_OJOS = 2
SGIE_ALERT_ID_BOCA = 3
SGIE_TOTAL_PERSON = 4
personas_alerta=set() # Para guardar solo una imagen por (persona, infracción)

current_obj_counter = {
    SGIE_TOTAL_PERSON:set(),
    SGIE_ALERT_ID_PIES:set(),
    SGIE_ALERT_ID_CABEZA:set(),
    SGIE_ALERT_ID_OJOS:set(),
    SGIE_ALERT_ID_BOCA:set()
    }
last_obj_counter = {
    SGIE_TOTAL_PERSON:set(),
    SGIE_ALERT_ID_PIES:set(),
    SGIE_ALERT_ID_CABEZA:set(),
    SGIE_ALERT_ID_OJOS:set(),
    SGIE_ALERT_ID_BOCA:set()
    }

pgie_classes_str= ["Vehicle", "TwoWheeler", "Person", "RoadSign"]
#sgie_classes_str= ["botas_acero", "casco", "gafas_seguridad", "tapabocas", "botas_caucho", "gafas_normal"]
#sgie_classes_str= ["botas_acero", "botas_caucho", "casco", "gafas_seguridad", "tapabocas", 
sgie_classes_str= ["pies_sin_proteccion", "cabeza_sin_proteccion", "ojos_sin_proteccion", "boca_sin_proteccion"]


config = configparser.ConfigParser()
config.read('riesgo-encubierto.ini')

date_str = str(datetime.now().strftime("%Y%m%d"))
logger = logging.getLogger('Log EPP RE')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(config['CONFIG']['log_path']+'/LogEPP_'+date_str+'.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p'))
logger.addHandler(fh)
sys.stderr.write = logger.warning
sys.stdout.write = logger.info

logger.info('INICIANDO EJECUCIÓN: RIESGO ENCUBIERTO - MODELO EPP - '+date_str)
logger.info('Archivo de Configuración -> riesgo_encubierto.ini : Ok')

save_path=config['GUARDAR_IMG']['save_path']
folder = config['GUARDAR_IMG']['folder_epp']+'/'+date_str
if not os.path.exists(save_path+'/'+folder) :
    os.makedirs(save_path+'/'+folder)
logger.info('Configuración Guardar imagenes Jetson: Ok')

pruebas = (config['CONFIG']['env'] == 'test')
logger.info('Ejecutando en ambiente: {}'.format(config['CONFIG']['env']))

# S3 bucket settings
s3_bucket_name = config['GUARDAR_IMG']['s3_bucket_name']
s3 = boto3.client('s3',
                    aws_access_key_id=config['GUARDAR_IMG']['s3_key_id'],
                    aws_secret_access_key=config['GUARDAR_IMG']['s3_access_key'],
                    config=Config(signature_version='s3v4')
                    )
logger.info('Configuración Guardar imagenes AWS S3: Ok')

# RDS settings
conn = pymysql.connect(host=config['BD']['rds_host'], user=config['BD']['user'], passwd=config['BD']['pwd'], 
db=config['BD']['db_name'], connect_timeout=int(config['BD']['timeout']))
model_id = config['BD']['model_id_epp']
logger.info('Configuración Base de Datos -> '+config['BD']['db_name']+': Ok')

# Función principal para definir la funcionalidad de lo que se va a hacer con cada deteccion
def tiler_src_pad_buffer_probe(pad,info,u_data,args):

    frame_number=0
    num_rects=0
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        logger.error("Unable to get GstBuffer ")
        return

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number=frame_meta.frame_num
        l_obj=frame_meta.obj_meta_list
        is_first_obj = True # Guarda el frame solo una vez así detecte varios objetos
        SAVE_IMAGE = False
        alerta = False
        obj_alerta = []
        fps = fps_streams["stream{0}".format(frame_meta.pad_index)].get_fps()

        while l_obj is not None:
            try: 
                obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            
            if obj_meta.unique_component_id == 1 : # pgie
                if obj_meta.class_id == PGIE_CLASS_ID_PERSON and validar_posicion(obj_meta.rect_params): # Si es persona, y esta en la entrada
                    obj_meta.rect_params.border_color.set(0.0, 0.0, 255.0, 1.0) #R,G,B,alpha
                    current_obj_counter[SGIE_TOTAL_PERSON].add(obj_meta.object_id)
                else :
                    obj_meta.rect_params.border_color.set(0.0, 0.0, 0.0, 0.0) #R,G,B,alpha
                #width=int(obj_meta.rect_params.width)
                #height=int(obj_meta.rect_params.height)
                #print('Ancho:',width, 'Alto:', height)
            elif obj_meta.unique_component_id == 2 and validar_posicion(obj_meta.parent.rect_params): # sgie y persona en la zona
                obj_meta.rect_params.border_color.set(255.0, 0.0, 0.0, 1.0) #R,G,B,alpha
                if (obj_meta.parent.object_id, obj_meta.class_id) not in personas_alerta : #and validar_alerta(obj_meta): 
                    #print('Alertas: ', personas_alerta)
                    obj_alerta.append(obj_meta) # Va agregando el EPP detectado, pintará varias veces el BB de la persona
                    personas_alerta.add((obj_meta.parent.object_id, obj_meta.class_id)) # Guarda persona/infraccion una sola vez
                    current_obj_counter[obj_meta.class_id].add(obj_meta.parent.object_id)
                    alerta = True
            
            if not pruebas :
                global start_time
                start_time = send_db_evento(current_obj_counter, start_time, fps)
            if alerta : 
                if is_first_obj:
                    is_first_obj = False
                    frame_image=get_frame(gst_buffer, frame_meta.batch_id)
                SAVE_IMAGE = True

            try: 
                l_obj=l_obj.next
            except StopIteration:
                break

        # Get frame rate through this probe
        fps_streams["stream{0}".format(frame_meta.pad_index)].get_fps()

        # Guardar la imagen
        if SAVE_IMAGE:
            frame_image=draw_bounding_boxes(frame_image,obj_alerta)
            file_name = folder+"/frame_"+str(frame_number)+".jpg"
            print('##### Guardando imagen: {}'.format(file_name))
            cv2.imwrite(save_path+"/"+file_name, frame_image)
            if not pruebas : # Solo sube a S3 en Produccion
                s3.upload_file(save_path+"/"+file_name, s3_bucket_name, file_name)
                send_db_alerta(file_name, obj_alerta)
            SAVE_IMAGE = False

        try:
            l_frame=l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK

# Esta función recibe SOLO los objetos en alerta y los marca en rojo sobre la imagen
def draw_bounding_boxes(image, obj_alerta) :
    for i, obj_meta in enumerate(obj_alerta) :
        rect_params=obj_meta.rect_params
        top=int(rect_params.top)
        left=int(rect_params.left)
        width=int(rect_params.width)
        height=int(rect_params.height)
        obj_name = 'Persona '+str(obj_meta.object_id)
        color = (255,0,0,0)
        
        if obj_meta.parent is not None : # EPP
            image = draw_bounding_boxes(image, [obj_meta.parent])
            obj_name=sgie_classes_str[obj_meta.class_id]
            #color = (0,255,0,0) if obj_meta.class_id in labels_epp_ok else (0,0,255,0)
            color = (0,0,255,0)
        #else :
        #    print('Ancho', width, 'Alto', height)

        if obj_meta.parent is None or pruebas : # Personas siempre y EPP pruebas
            #print('{}: left:{}, top:{}, width:{}, height:{}'.format(obj_name, left, top, width, height))
            image=cv2.rectangle(image,(left,top),(left+width,top+height),color,1)
            image=cv2.putText(image,obj_name,(left+5,top+height-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1)
        else : # EPP Prod (Solo tags)
            image=cv2.putText(image,obj_name, (10, TILED_OUTPUT_HEIGHT-10-15*i), cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1)
    return image

# Esta funcion valida la posicion de la persona en la entrada
# Esto hay que volverlo a configurar dependiendo de la zona donde este la camara
# 2021-08-20: tercera parte central, por debajo de la tercera parte de la pantalla
# 2021-09-21: Cuarta parte (2x2) inferior-izquierda
def validar_posicion(rect_params) :
    top_persona=int(rect_params.top)
    left_persona=int(rect_params.left)
    width_persona=int(rect_params.width)
    height_persona=int(rect_params.height)
    base_persona = top_persona+height_persona
    return base_persona > TILED_OUTPUT_HEIGHT/2 and left_persona+width_persona < TILED_OUTPUT_WIDTH/2

# Esta función valida la posición del EPP respecto al cuerpo de la persona :
def validar_alerta(obj_meta) :
    rect_params=obj_meta.parent.rect_params
    top_persona=int(rect_params.top)
    height_persona=int(rect_params.height)
    base_persona = top_persona+height_persona

    alerta = False
    rect_params_epp=obj_meta.rect_params
    if obj_meta.class_id == 0 : # Pies
        if int(rect_params_epp.top) >= base_persona*0.8 : # Las botas están por debajo del 80% del cuerpo
            alerta = True
    else :
        if int(rect_params_epp.top) <= base_persona*0.2 : # Otro EPP (cara) están por encima del 20% del cuerpo
            alerta = True
    return alerta


# Esta función almacena datos en la BD cada intervalo de tiempo configurado en el .ini
def send_db_evento(obj_counter, st_time, fps) :
    end_time=time.time()
    intervalo = end_time-st_time
    if(intervalo > int(config['BD']['interval_sec'])):
        hora = time.gmtime(end_time).tm_hour
        turno = 3 if hora in range(5,17) else 2
        personas_total = len(obj_counter[SGIE_TOTAL_PERSON] - last_obj_counter[SGIE_TOTAL_PERSON])
        if (personas_total > 0) :
            print("GUARDANDO EN BD: {}".format(config['BD']['db_name']))
            with conn.cursor() as cur:
                cur.execute('INSERT INTO Events (description, fk_idTurn, personsPerEvent, vehiclePerEvent, idRiskModule, date) \
                            VALUES ({}, {}, {}, {}, {}, {})'.format("'Evento EPP desde Jetson'", turno, personas_total, 0, model_id, "now()"))
                fk_id = cur.lastrowid
                for i in range(0,4) :
                    total = len(obj_counter[i] - last_obj_counter[i])
                    cur.execute('INSERT INTO EventsTypeViolation (fk_idEvent, fk_idTypeViolation, countElements)  \
                                VALUES ({}, {}, {})'.format(fk_id, i+3, total))
                    last_obj_counter[i] = obj_counter[i].copy()
                conn.commit()
            last_obj_counter[SGIE_TOTAL_PERSON] = obj_counter[SGIE_TOTAL_PERSON].copy()

        st_time=end_time
    return st_time

# Esta función almacena datos en la BD cada intervalo de tiempo configurado en el .ini
def send_db_alerta(file_name, obj_alerta) :
    tags = set([obj.class_id for obj in obj_alerta])
    with conn.cursor() as cur:
        cur.execute('INSERT INTO Evidences (description, url, idRiskModule, date) \
                    VALUES ({}, {}, {}, {})'.format("'Alerta EPP desde Jetson'", "'"+file_name+"'", model_id, "now()"))
        fk_id = cur.lastrowid
        for tag in tags :
            cur.execute('INSERT INTO EvidencesTypeViolation (fk_idEvidence, fk_idTypeViolation) \
                        VALUES ({}, {})'.format(fk_id, tag+3))
        conn.commit()


def get_frame(gst_buffer, batch_id):
    n_frame=pyds.get_nvds_buf_surface(hash(gst_buffer),batch_id)
    frame_image=np.array(n_frame,copy=True,order='C')
    frame_image=cv2.cvtColor(frame_image,cv2.COLOR_RGBA2BGRA)
    #frame_image = jetson.utils.cudaFromNumpy(frame_image)
    return frame_image

def cb_newpad(decodebin, decoder_src_pad,data):
    print("In cb_newpad")
    caps=decoder_src_pad.get_current_caps()
    gststruct=caps.get_structure(0)
    gstname=gststruct.get_name()
    source_bin=data
    features=caps.get_features(0)

    print("gstname=",gstname)
    if(gstname.find("video")!=-1):
        print("features=",features)
        if features.contains("memory:NVMM"):
            # Get the source bin ghost pad
            bin_ghost_pad=source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write("Failed to link decoder src pad to source bin ghost pad")
        else:
            sys.stderr.write(" Error: Decodebin did not pick nvidia decoder plugin.")

def decodebin_child_added(child_proxy,Object,name,user_data):
    print("Decodebin child added:", name)
    if(name.find("decodebin") != -1):
        Object.connect("child-added",decodebin_child_added,user_data)   
    # Se elimino el bufapi-version

def create_source_bin(index,uri):
    print("Creating source bin")

    bin_name="source-bin-%02d" %index
    print(bin_name)
    nbin=Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write(" Unable to create source bin ")

    uri_decode_bin=Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    if not uri_decode_bin:
        sys.stderr.write(" Unable to create uri decode bin ")
    uri_decode_bin.set_property("uri",uri)
    uri_decode_bin.connect("pad-added",cb_newpad,nbin)
    uri_decode_bin.connect("child-added",decodebin_child_added,nbin)

    Gst.Bin.add(nbin,uri_decode_bin)
    bin_pad=nbin.add_pad(Gst.GhostPad.new_no_target("src",Gst.PadDirection.SRC))
    if not bin_pad:
        sys.stderr.write(" Failed to add ghost pad in source bin ")
        return None
    return nbin

def main(args):
    number_sources = len(args)-1
    if number_sources == 0 :
        number_sources = 1

    for i in range(0,number_sources):
        fps_streams["stream{0}".format(i)]=GETFPS(i)
    
    GObject.threads_init()
    Gst.init(None)

    # Create gstreamer elements */
    print("Creating Pipeline  ")
    pipeline = Gst.Pipeline()
    is_live = False

    # Creacion del pipeline principal
    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline ")
    print("Creating streamux  ")

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux ")

    pipeline.add(streammux)
    for i in range(number_sources):
        print("Creating source_bin ",i)
        uri_name = args[i+1] if len(args) > 1 else config['CONFIG']['uri_cam1']
        if uri_name.find("rtsp://") == 0 :
            is_live = True
            print('LIVE', uri_name)

        source_bin=create_source_bin(i, uri_name)
        if not source_bin:
            sys.stderr.write("Unable to create source bin ")
        pipeline.add(source_bin)
        padname="sink_%u" %i
        sinkpad= streammux.get_request_pad(padname) 
        if not sinkpad:
            sys.stderr.write("Unable to create sink pad bin ")
        srcpad=source_bin.get_static_pad("src")
        if not srcpad:
            sys.stderr.write("Unable to create src pad bin ")
        srcpad.link(sinkpad)

    queue1=Gst.ElementFactory.make("queue","queue1")
    queue2=Gst.ElementFactory.make("queue","queue2")
    queue3=Gst.ElementFactory.make("queue","queue3")
    queue4=Gst.ElementFactory.make("queue","queue4")
    queue5=Gst.ElementFactory.make("queue","queue5")
    queue6=Gst.ElementFactory.make("queue","queue6")
    queue7=Gst.ElementFactory.make("queue","queue7")
    pipeline.add(queue1)
    pipeline.add(queue2)
    pipeline.add(queue3)
    pipeline.add(queue4)
    pipeline.add(queue5)
    pipeline.add(queue6)
    pipeline.add(queue7)
    
    print("Creating Pgie  ")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie ")
    print("Creating tiler  ")

    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    if not tracker:
        sys.stderr.write(" Unable to create tracker ")

    sgie1 = Gst.ElementFactory.make("nvinfer", "secondary1-nvinference-engine")
    if not sgie1:
        sys.stderr.write(" Unable to make sgie1 ")

    print("Creating nvvidconv1  ")
    nvvidconv1 = Gst.ElementFactory.make("nvvideoconvert", "convertor1")
    if not nvvidconv1:
        sys.stderr.write(" Unable to create nvvidconv1 ")

    print("Creating filter1  ")
    caps1 = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
    filter1 = Gst.ElementFactory.make("capsfilter", "filter1")
    if not filter1:
        sys.stderr.write(" Unable to get the caps filter1 ")
    filter1.set_property("caps", caps1)

    print("Creating tiler  ")
    tiler=Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    if not tiler:
        sys.stderr.write(" Unable to create tiler ")

    print("Creating nvvidconv  ")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write(" Unable to create nvvidconv ")

    print("Creating nvosd  ")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    if not nvosd:
        sys.stderr.write(" Unable to create nvosd ")
    nvosd.set_property('process-mode',OSD_PROCESS_MODE)
    nvosd.set_property('display-text',OSD_DISPLAY_TEXT)

    print("Creating EGLSink")
    if pruebas :
        transform=Gst.ElementFactory.make("nvegltransform", "nvegl-transform")
        sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
        print("Executing on Test env ")
    else :
        sink = Gst.ElementFactory.make("fakesink", "fakesink")
        print("Executing on env:", config['CONFIG']['env'])
    if not sink:
        sys.stderr.write(" Unable to create egl sink ")

    if is_live:
        print("Atleast one of the sources is live")
        streammux.set_property('live-source', 1)

    streammux.set_property('width', MUXER_OUTPUT_WIDTH)
    streammux.set_property('height', MUXER_OUTPUT_HEIGHT)
    streammux.set_property('batch-size', number_sources)
    streammux.set_property('batched-push-timeout', 4000000)

    #pgie.set_property('config-file-path', "trafficamnet_config.txt")
    pgie.set_property('config-file-path', config['MODELOS']['config_file'])
    pgie_batch_size=pgie.get_property("batch-size")
    if(pgie_batch_size != number_sources):
        logger.warning("WARNING: Overriding infer-config batch-size",pgie_batch_size," with number of sources ", number_sources)
        pgie.set_property("batch-size",number_sources)

    # Set properties of tracker
    tracker_config = configparser.ConfigParser()
    tracker_config.read(config['TRACKER']['tracker_config_file'])

    for key in tracker_config['tracker']:
        if key == 'tracker-width':
            tracker_width = tracker_config.getint('tracker', key)
            tracker.set_property('tracker-width', tracker_width)
        if key == 'tracker-height':
            tracker_height = tracker_config.getint('tracker', key)
            tracker.set_property('tracker-height', tracker_height)
        if key == 'gpu-id':
            tracker_gpu_id = tracker_config.getint('tracker', key)
            tracker.set_property('gpu_id', tracker_gpu_id)
        if key == 'll-lib-file':
            tracker_ll_lib_file = tracker_config.get('tracker', key)
            tracker.set_property('ll-lib-file', tracker_ll_lib_file)
        if key == 'll-config-file':
            tracker_ll_config_file = tracker_config.get('tracker', key)
            tracker.set_property('ll-config-file', tracker_ll_config_file)
        if key == 'enable-batch-process':
            tracker_enable_batch_process = tracker_config.getint('tracker', key)
            tracker.set_property('enable_batch_process',
                                 tracker_enable_batch_process)

    #sgie1.set_property('config-file-path', "lpd_us_config.txt")
    sgie1.set_property('config-file-path', config['MODELOS']['epp_config_file'])
    sgie1.set_property('process-mode', 2)

    tiler_rows=int(math.sqrt(number_sources))
    tiler_columns=int(math.ceil((1.0*number_sources)/tiler_rows))
    tiler.set_property("rows",tiler_rows)
    tiler.set_property("columns",tiler_columns)
    tiler.set_property("width", TILED_OUTPUT_WIDTH)
    tiler.set_property("height", TILED_OUTPUT_HEIGHT)
    sink.set_property("qos",0)
    sink.set_property("sync",0)

    print("Adding elements to Pipeline ")    
    pipeline.add(pgie)
    pipeline.add(tracker)
    pipeline.add(sgie1)
    pipeline.add(tiler)
    pipeline.add(nvvidconv)
    pipeline.add(filter1)
    pipeline.add(nvvidconv1)
    pipeline.add(nvosd)
    if pruebas :
        pipeline.add(transform)
    pipeline.add(sink)

    print("Linking elements in the Pipeline ")
    streammux.link(pgie)
    pgie.link(queue1)
    queue1.link(tracker)
    tracker.link(queue2)
    queue2.link(sgie1)
    sgie1.link(queue3)
    queue3.link(nvvidconv1)
    nvvidconv1.link(queue4)
    queue4.link(filter1)
    filter1.link(queue5)
    queue5.link(tiler)
    tiler.link(queue6)
    queue6.link(nvvidconv)
    nvvidconv.link(queue7)
    queue7.link(nvosd)
    if pruebas:
        nvosd.link(transform)
        transform.link(sink)
    else:
        nvosd.link(sink)

    # create an event loop and feed gstreamer bus mesages to it
    loop = GObject.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect ("message", bus_call, loop)

    tiler_src_pad=tiler.get_static_pad("sink")
    if not tiler_src_pad:
        sys.stderr.write(" Unable to get sink pad ")
    else:
        tiler_src_pad.add_probe(Gst.PadProbeType.BUFFER, tiler_src_pad_buffer_probe, 0, args)

    # List the sources
    print("Now playing...")
    for i, source in enumerate(args):
        if ((len(args) - 1) > i > 3):
            print(i, ": ", source)

    print("Starting pipeline ")
    # start play back and listed to events		
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    # cleanup
    thread_stop = True
    print("Exiting app")
    pipeline.set_state(Gst.State.NULL)

if __name__ == '__main__':
    sys.exit(main(sys.argv))
