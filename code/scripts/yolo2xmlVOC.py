import xml.etree.ElementTree as ET
import xml.dom.minidom
import glob
import cv2


class_names = ['complete', 'incomplete']

def get_param_from_yolo(img_file):
    path = "/".join(img_file.split('/')[0:-1])
    txt_file = img_file.split('/')[-1].split('.')[0]
    img_name = img_file.split('/')[-1]
    txt_file = path + '/' + txt_file + '.txt'

    img = cv2.imread(img_file)
    img_h, img_w, img_d = img.shape

    with open(txt_file, 'r') as f:
        data = f.readlines()[0].split(' ')
        class_id = int(data[0])
        class_name = class_names[class_id]
        x_c = float(data[1]) * img_w
        y_c = float(data[2]) * img_h
        w = float(data[3]) * img_w
        h = float(data[4]) * img_h
        x_min = int(x_c - 0.5 * w)
        y_min = int(y_c - 0.5 * h)
        x_max = int(x_min + w)
        y_max = int(y_min + h)
    return ['images', img_name, img_w, img_h, img_d, class_name, x_min, y_min, x_max, y_max]

def write_xml_data(data, path):
    node_root = ET.Element('annotation')
    node_folder = ET.SubElement(node_root, 'folder')
    node_folder.text = data[0]

    node_filename = ET.SubElement(node_root, 'filename')
    node_filename.text = data[1]
    img_name = data[1].split('.')[0]
    # node_source= SubElement(node_root, 'source')
    # node_database = SubElement(node_source, 'database')
    # node_database.text = 'Coco database'
    
    node_size = ET.SubElement(node_root, 'size')
    node_width = ET.SubElement(node_size, 'width')
    node_width.text = str(data[2])

    node_height = ET.SubElement(node_size, 'height')
    node_height.text = str(data[3])

    node_depth = ET.SubElement(node_size, 'depth')
    node_depth.text = str(data[4])

    node_segmented = ET.SubElement(node_root, 'segmented')
    node_segmented.text = '0'


    node_object = ET.SubElement(node_root, 'object')
    node_name = ET.SubElement(node_object, 'name')
    node_name.text = str(data[5])
    
    node_pose = ET.SubElement(node_object, 'pose')
    node_pose.text = 'Unspecified'
    
    
    node_truncated = ET.SubElement(node_object, 'truncated')
    node_truncated.text = '0'
    node_difficult = ET.SubElement(node_object, 'difficult')
    node_difficult.text = '0'
    node_bndbox = ET.SubElement(node_object, 'bndbox')
    node_xmin = ET.SubElement(node_bndbox, 'xmin')
    node_xmin.text = str(data[6])
    node_ymin = ET.SubElement(node_bndbox, 'ymin')
    node_ymin.text = str(data[7])
    node_xmax = ET.SubElement(node_bndbox, 'xmax')
    node_xmax.text =  str(data[8])
    node_ymax = ET.SubElement(node_bndbox, 'ymax')
    node_ymax.text = str(data[9])

    xmlstr = ET.tostring(node_root, encoding='unicode', method='xml')

    xmlstr = xml.dom.minidom.parseString(xmlstr).toprettyxml()
    f = open(f'{path}/{img_name}.xml', 'w')
    f.write(xmlstr)
    f.close()

def convert_file(f):
    path = f.split('/')[0:-1]
    path = '/'.join(path)
    data = get_param_from_yolo(f)
    write_xml_data(data, path)

def start():
    path = '../../volumes/added_background/incomplete'
    for f in glob.glob(f"{path}/*.jpg"):
        convert_file(f)
    print('done')


start()
