import src.utils.convert_xml as xml
import xmltodict
import dicttoxml

dic = {'camera_001': {
    'gauge_001': {'size': '100', 'color': 'red'},
    'gauge_002': {'size': '200', 'color': 'blue'},
    'gauge_003': {'size': '300', 'color': 'green'},
}}


xml_text = xmltodict.unparse(dic, pretty=True)

with open('test.xml', 'w') as f:
    f.write(xml_text)

with open('test.xml', 'r') as f:
    dic2 = f.read()

print(xmltodict.parse(dic2))

