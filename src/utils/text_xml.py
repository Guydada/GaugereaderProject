import src.utils.convert_xml as xml
import xmltodict
import dicttoxml

dic = {'camera_001': {
    'gauge_001': {'size': '100', 'color': 'red'},
    'gauge_002': {'size': '200', 'color': 'blue'},
    'gauge_003': {'size': '300', 'color': 'green'},
}}

dic2 = xmltodict.parse(xmltodict.unparse(dic, pretty=True))
print(dic2)
print(dic2['camera_001']['gauge_001']['size'])