import os
import xmltodict


def dict_to_xml(data: dict,
                path: str,
                gauge: bool = False) -> str:
    """
    Convert a dictionary to XML.

    :param gauge:
    :param path:
    :param data:
    :return: the XML string
    """
    if gauge:
        dic = {'gauge': data}
    else:
        dic = data
    xml = xmltodict.unparse(dic, pretty=True)
    with open(path, 'w') as f:
        f.write(xml)
    return str(path)


def xml_to_dict(path: str,
                gauge: bool = False) -> dict:
    """
    Convert an XML file to a dictionary.

    :param gauge:
    :param path: the path to the XML file
    :return: the dictionary
    """
    with open(path, 'r') as xml_object:
        xml_dict = xmltodict.parse(xml_object.read())
    if gauge:
        return xml_dict['gauge']
    return xml_dict


def dict_append_to_xml(data: dict,
                       path: str) -> str:
    """
    Append a dictionary to an XML file.

    :param path: the path to the XML file
    :param data: the dictionary
    :return: the path to the XML file
    """
    xml_dict = xml_to_dict(path)
    xml_dict.update(data)
    return dict_to_xml(xml_dict, path)
