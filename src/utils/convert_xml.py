import os
import xmltodict


def dict_to_xml(data: dict,
                path: str) -> str:
    """
    Convert a dictionary to XML.

    :param path:
    :param data:
    :return: the XML string
    """
    xml = xmltodict.unparse(data, pretty=True)
    with open(path, 'wb') as f:
        f.write(xml)
    return str(path)


def xml_to_dict(path: str) -> dict:
    """
    Convert an XML file to a dictionary.

    :param path: the path to the XML file
    :return: the dictionary
    """
    with open(path, 'r') as xml_object:
        xml_dict = xmltodict.parse(xml_object.read())
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
