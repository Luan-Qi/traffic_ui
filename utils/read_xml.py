import xml.etree.ElementTree as ET
import numpy as np
import ast

def convert_value(text):
    """将XML文本转换为合适的Python数据类型"""
    try:
        return ast.literal_eval(text.strip())
    except (ValueError, SyntaxError, AttributeError):
        try:
            return int(text)
        except (ValueError, TypeError):
            try:
                return float(text)
            except (ValueError, TypeError):
                return text


def convert_value_with_numpy(text):
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError, AttributeError):
        pass

    # 尝试转换为int
    try:
        return int(text)
    except (ValueError, TypeError):
        pass

    # 尝试转换为float
    try:
        return float(text)
    except (ValueError, TypeError):
        pass

    stripped_text = text.strip()
    if stripped_text.startswith('[') and stripped_text.endswith(']'):
        try:
            # 移除方括号并转换
            array = np.fromstring(stripped_text[1:-1], sep=' ')
            return array
        except (ValueError, TypeError):
            pass

    # 如果所有转换都失败，则返回原始文本
    return stripped_text


def parse_element(element):
    """递归解析XML元素"""
    data = {}
    for child in element:
        # 递归处理子元素
        if len(child) > 0 or child.attrib:
            value = parse_element(child)
        else:
            value = convert_value_with_numpy(child.text)

        # 处理重复键（将值转换为列表）
        if child.tag in data:
            if not isinstance(data[child.tag], list):
                data[child.tag] = [data[child.tag]]
            data[child.tag].append(value)
        else:
            data[child.tag] = value
    return data


def xml_to_dict_for_dir(xml_file):
    """将XML文件转换为字典"""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    dir_data = root.find('dir')

    result = {}
    for dir_element in dir_data.findall('dir'):
        key = convert_value(dir_element.get('key'))
        result[key] = parse_element(dir_element)
    return result

if __name__ == '__main__':
    data_dict = xml_to_dict_for_dir(r'../output/DJI_0013_lane.xml')
    print(data_dict)