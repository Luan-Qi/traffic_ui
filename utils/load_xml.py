import math
import xml.etree.ElementTree as ET

import numpy as np

import xml.etree.ElementTree as ET
import numpy as np


def read_xml(input_path):
    tree = ET.parse(input_path)
    root = tree.getroot()

    routes = root.find('routes')
    lanes = root.find('lanes')
    connectors = root.find('connectors')
    nodes = root.find('nodes')

    dir = {}
    for route in routes.findall('route'):
        dir_id = int(route.get('roadNo')) - 1
        if(dir_id not in dir):
            dir[dir_id] = {}
        dir[dir_id]['direction'] = 5 - int(route.get('roadLevel'))
        dir[dir_id]['laneNum'] = int(route.get('laneNum'))
        dir[dir_id]['length'] = float(route.get('length'))
        dir[dir_id]['width'] = float(route.get('width'))
        dir[dir_id]['upNodeId'] = int(route.get('upNodeId'))
        dir[dir_id]['downNodeId'] = int(route.get('downNodeId'))
        dir[dir_id]['geom'] = [float(i) for i in route.get('geom').split(', ')]
        dir[dir_id]['type'] = int(route.get('type'))
        dir[dir_id]['typeId'] = int(route.get('typeId'))
        dir[dir_id]['priority'] = int(route.get('priority'))
        dir[dir_id]['spreadType'] = route.get('spreadType')
        dir[dir_id]['endOffset'] = float(route.get('endOffset'))
        dir[dir_id]['distance'] = float(route.get('distance'))
        dir[dir_id]['createTime'] = route.get('createTime')
        dir[dir_id]['updateTime'] = route.get('updateTime')
        dir[dir_id]['bikeLaneWidth'] = float(route.get('bikeLaneWidth'))

    for lane in lanes.findall('lane'):
        dir_id = int(lane.get('roadsegId')) - 1
        lane_id = int(lane.get('laneNo'))
        while len(dir) <= dir_id:
            dir.append({})
        if 'Line' + str(lane_id) not in dir[dir_id]:
            dir[dir_id]['Line' + str(lane_id)] = {}
        dir[dir_id]['Line' + str(lane_id)]['L1'] = {}
        dir[dir_id]['Line' + str(lane_id)]['L1']['adjacentNodeId'] = int(lane.get('adjacentNodeId'))
        dir[dir_id]['Line' + str(lane_id)]['L1']['detectorId'] = int(lane.get('detectorId'))
        dir[dir_id]['Line' + str(lane_id)]['L1']['direction'] = int(lane.get('direction'))
        dir[dir_id]['Line' + str(lane_id)]['L1']['nodeId'] = int(lane.get('nodeId'))
        dir[dir_id]['Line' + str(lane_id)]['L1']['isLocateLamp'] = lane.get('isLocateLamp') == 'true'
        dir[dir_id]['Line' + str(lane_id)]['L1']['isMixedLane'] = lane.get('isMixedLane') == 'true'
        dir[dir_id]['Line' + str(lane_id)]['L1']['isRightSpecial'] = lane.get('isRightSpecial') == 'true'
        dir[dir_id]['Line' + str(lane_id)]['L1']['acceleration'] = lane.get('acceleration') == 'true'
        dir[dir_id]['Line' + str(lane_id)]['L1']['endOffset'] = float(lane.get('endOffset'))
        dir[dir_id]['Line' + str(lane_id)]['L1']['speed'] = float(lane.get('speed'))
        dir[dir_id]['Line' + str(lane_id)]['L1']['width'] = float(lane.get('width'))
        dir[dir_id]['Line' + str(lane_id)]['L1']['leftRealLineLen'] = float(lane.get('leftRealLineLen'))
        dir[dir_id]['Line' + str(lane_id)]['L1']['rightRealLineLen'] = float(lane.get('rightRealLineLen'))

    for connector in connectors.findall('connector'):
        fromEdge = int(connector.get('fromEdge')) - 1
        fromLane = int(connector.get('fromLane'))
        toEdge = int(connector.get('toEdge')) - 1
        toLane = int(connector.get('toLane'))
        shape = [float(i) for i in connector.get('shape').split(', ')]
        # 根据需要处理连接器信息

    return dir


def load_routes(xml_path):
    """
    从XML文件中载入routes信息
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    routes_info = []
    routes_elem = root.find('routes')
    if routes_elem is not None:
        for route_elem in routes_elem.findall('route'):
            route_info = {
                'roadNo': route_elem.get('roadNo'),
                'capacity': route_elem.get('capacity'),
                'name': route_elem.get('name'),
                'desp': route_elem.get('desp'),
                'laneNum': route_elem.get('laneNum'),
                'length': route_elem.get('length'),
                'width': route_elem.get('width'),
                'roadLevel': route_elem.get('roadLevel'),
                'type': route_elem.get('type'),
                'upNodeId': route_elem.get('upNodeId'),
                'downNodeId': route_elem.get('downNodeId'),
                'typeId': route_elem.get('typeId'),
                'priority': route_elem.get('priority'),
                'geom': route_elem.get('geom'),
                'spreadType': route_elem.get('spreadType'),
                'endOffset': route_elem.get('endOffset'),
                'distance': route_elem.get('distance'),
                'createTime': route_elem.get('createTime'),
                'updateTime': route_elem.get('updateTime'),
                'bikeLaneWidth': route_elem.get('bikeLaneWidth')
            }
            routes_info.append(route_info)
    return routes_info


def load_lanes(xml_path,dir):
    """
    从XML文件中载入lanes信息
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    lanes_elem = root.find('lanes')
    if lanes_elem is not None:
        for lane_elem in lanes_elem.findall('lane'):
            dir_id = int(lane_elem.get('roadsegId')) - 1
            lane_id = int(lane_elem.get('laneNo'))
            if(dir_id not in dir):
                dir[dir_id] = {}
            if(lane_id not in dir[dir_id]):
                dir[dir_id][lane_id] = {}
            direction_l = 5 - int(lane_elem.get('direction'))
            dir[dir_id][lane_id] = {
                'others':{
                    'adjacentNodeId': lane_elem.get('adjacentNodeId'),
                    'detectorId': lane_elem.get('detectorId'),
                    'nodeId': lane_elem.get('nodeId'),
                    'isLocateLamp': lane_elem.get('isLocateLamp'),
                    'isMixedLane': lane_elem.get('isMixedLane'),
                    'isRightSpecial': lane_elem.get('isRightSpecial'),
                    'acceleration': lane_elem.get('acceleration'),
                    'endOffset': lane_elem.get('endOffset'),
                    'speed': lane_elem.get('speed'), # 限速
                    'width': lane_elem.get('width'),
                    'leftRealLineLen': lane_elem.get('leftRealLineLen'),
                    'rightRealLineLen': lane_elem.get('rightRealLineLen')
                },

                'direction': direction_l,


            }
    return dir

def load_connectors(xml_path):
    """
    从XML文件中载入connectors信息
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    connectors_info = []
    connectors_elem = root.find('connectors')
    if connectors_elem is not None:
        for connector_elem in connectors_elem.findall('connector'):
            connector_info = {
                'nodeId': connector_elem.get('nodeId'),
                'num': connector_elem.get('num'),
                'contpos': connector_elem.get('contpos'),
                'fromEdge': connector_elem.get('fromEdge'),
                'fromLane': connector_elem.get('fromLane'),
                'indirect': connector_elem.get('indirect'),
                'keepClear': connector_elem.get('keepClear'),
                'length': connector_elem.get('length'),
                'isLeft': connector_elem.get('isLeft'),
                'leftlength': connector_elem.get('leftlength'),
                'pass': connector_elem.get('pass'),
                'speed': connector_elem.get('speed'),
                'toEdge': connector_elem.get('toEdge'),
                'toLane': connector_elem.get('toLane'),
                'uncontrolled': connector_elem.get('uncontrolled'),
                'visibility': connector_elem.get('visibility'),
                'linkIndex': connector_elem.get('linkIndex'),
                'shape': connector_elem.get('shape')
            }
            connectors_info.append(connector_info)
    return connectors_info

def load_nodes(xml_path):
    """
    从XML文件中载入nodes信息
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    nodes_info = []
    nodes_elem = root.find('nodes')
    if nodes_elem is not None:
        for node_elem in nodes_elem.findall('node'):
            node_info = {
                'num': node_elem.get('num'),
                'directionNum': node_elem.get('directionNum'),
                'latitude': node_elem.get('latitude'),
                'longitude': node_elem.get('longitude'),
                'keepClear': node_elem.get('keepClear'),
                'radius': node_elem.get('radius'),
                'tl': node_elem.get('tl'),
                'x': node_elem.get('x'),
                'y': node_elem.get('y'),
                'z': node_elem.get('z'),
                'createTime': node_elem.get('createTime'),
                'updateTime': node_elem.get('updateTime'),
                'fringe': node_elem.get('fringe'),
                'rightOfWay': node_elem.get('rightOfWay'),
                'tlLayout': node_elem.get('tlLayout'),
                'tlType': node_elem.get('tlType'),
                'type': node_elem.get('type')
            }
            nodes_info.append(node_info)
    return nodes_info


def write_roads(dir,scale,outputpath):
    def transform(points, scale = scale):
        # 坐标转换，像素到实际
        return [str(round(point * scale,1)) for point in points]
    # 创建根元素和子元素
    root = ET.Element('root')
    child1 = ET.SubElement(root, 'routes')
    child2 = ET.SubElement(root, 'lanes')
    for dir_id in range(len(dir)):
        direction = dir[dir_id]['direction']
        direction_write = str(5 - direction)
        points_start = dir[dir_id]['stop']['points']
        x1 = (points_start[0][0] + points_start[1][0]) / 2
        y1 = (points_start[0][1] + points_start[1][1]) / 2

        points_ends = []
        RealLineLens = []
        for line_id in range(len(dir[dir_id])-6):
            lines_start = dir[dir_id]['Line' + str(line_id)]['L1']['points'][0]
            lines_end = dir[dir_id]['Line' + str(line_id)]['L1']['points'][1]
            RealLineLen = round(math.sqrt((lines_start[0] - lines_end[0])**2 + (lines_start[1] - lines_end[1])**2) * scale,1)
            points_ends.append(lines_end)
            points_ends.append(lines_end)
            RealLineLens.append(RealLineLen)
        length = max(RealLineLens)
        x2 = (points_ends[0][0] + points_ends[-1][0]) / 2
        y2 = (points_ends[0][1] + points_ends[-1][1]) / 2
        points = transform([x1,y1,x2,y2],scale)
        points_write = ', '.join(points)
        nodes = write_routes(child1,dir_id,len(dir[dir_id])-7,length,points_write,mode='no')  # 不需要两端路口

        for lane_id in range(1,len(dir[dir_id])-6):
            write_lanes(child2,dir_id,lane_id,nodes,direction_write,RealLineLens[lane_id-1],RealLineLens[lane_id])



def write_crosses(dir,rules,out_num,scale,outputpath):
    def transform(points, scale=scale):
        # 坐标转换，像素到实际
        return [str(round(point * scale, 1)) for point in points]

    direction2roadNo ={}
    # 反向route的坐标
    route_out = {}
    # 创建根元素和子元素
    root = ET.Element('root')
    child1 = ET.SubElement(root, 'routes')
    child2 = ET.SubElement(root, 'lanes')
    for dir_id in range(len(dir)):
        route_out[dir_id+1+len(dir)] = {}
        direction = dir[dir_id]['direction']
        direction_write = str(5 - direction)
        direction2roadNo[direction] = dir_id +1
        points_start = dir[dir_id]['stop']['points']
        x2 = (points_start[0][0] + points_start[1][0]) / 2
        y2 = (points_start[0][1] + points_start[1][1]) / 2

        points_ends = []
        RealLineLens = []
        for line_id in range(len(dir[dir_id]) - 6):
            lines_start = dir[dir_id]['Line' + str(line_id)]['L1']['points'][0]
            lines_end = dir[dir_id]['Line' + str(line_id)]['L1']['points'][1]
            RealLineLen = round(math.sqrt((lines_start[0] - lines_end[0]) ** 2 + (lines_start[1] - lines_end[1]) ** 2) * scale, 1)
            points_ends.append(lines_end)
            RealLineLens.append(RealLineLen)
        length = max(RealLineLens)
        x1 = (points_ends[0][0] + points_ends[-1][0]) / 2
        y1 = (points_ends[0][1] + points_ends[-1][1]) / 2
        points = transform([x1, y1, x2, y2], scale)
        points_write = ', '.join(points)
        nodes = write_routes(child1, dir_id, len(dir[dir_id]) - 7, length, points_write, mode='in')  # 进口道

        for lane_id in range(1, len(dir[dir_id]) - 6):
            # write_lanes(child2, dir_id, lane_id, nodes, direction_write, RealLineLens[lane_id - 1],RealLineLens[lane_id])
            write_lanes(child2, dir_id, lane_id, nodes, direction_write, length,length)

        # 写与该进口道反方向的出口道
        fit_stop = dir[dir_id]['stop']['fit']
        reverse_lane_num = out_num[direction]
        move_dis = (len(dir[dir_id]) - 5 + reverse_lane_num)*3.5/2
        if direction <= 2:
            x2_re = max(round(x1*scale + move_dis / math.sqrt(fit_stop[0] ** 2 + 1),1),0)
            y2_re = max(round(y1*scale + abs(fit_stop[0]) * move_dis / math.sqrt(fit_stop[0] ** 2 + 1),1),0)
            x1_re = max(round(x2*scale + move_dis / math.sqrt(fit_stop[0] ** 2 + 1),1),0)
            y1_re = max(round(y2*scale + abs(fit_stop[0]) * move_dis / math.sqrt(fit_stop[0] ** 2 + 1),1),0)
            direction_reverse_write = str(3 - direction)
        else:
            x2_re = max(round(x1*scale - move_dis / math.sqrt(fit_stop[0] ** 2 + 1),1),0)
            y2_re = max(round(y1*scale - abs(fit_stop[0]) * move_dis / math.sqrt(fit_stop[0] ** 2 + 1),1),0)
            x1_re = max(round(x2*scale - move_dis / math.sqrt(fit_stop[0] ** 2 + 1),1),0)
            y1_re = max(round(y2*scale - abs(fit_stop[0]) * move_dis / math.sqrt(fit_stop[0] ** 2 + 1),1),0)
            direction_reverse_write = str(7 - direction)
        points_write = ', '.join([str(x1_re),str(y1_re),str(x2_re),str(y2_re)])
        nodes = write_routes(child1, dir_id+len(dir), reverse_lane_num, length, points_write, mode='out')  # 进口道

        # 该route反方向最左侧坐标
        move_dis = (len(dir[dir_id]) - 5) * 3.5 / 2
        if direction <= 2:
            out_left_x2_re = max(round(x1*scale + move_dis / (math.sqrt(fit_stop[0] ** 2 + 1)),1),0)
            out_left_y2_re = max(round(y1*scale + fit_stop[0] * move_dis / math.sqrt(fit_stop[0] ** 2 + 1),1),0)
            out_left_x1_re = max(round(x2*scale + move_dis / math.sqrt(fit_stop[0] ** 2 + 1),1),0)
            out_left_y1_re = max(round(y2*scale + fit_stop[0] * move_dis / math.sqrt(fit_stop[0] ** 2 + 1),1),0)
        else:
            out_left_x2_re = max(round(x1*scale - move_dis / math.sqrt(fit_stop[0] ** 2 + 1),1),0)
            out_left_y2_re = max(round(y1*scale - fit_stop[0] * move_dis / math.sqrt(fit_stop[0] ** 2 + 1),1),0)
            out_left_x1_re = max(round(x2*scale - move_dis / math.sqrt(fit_stop[0] ** 2 + 1),1),0)
            out_left_y1_re = max(round(y2*scale - fit_stop[0] * move_dis / math.sqrt(fit_stop[0] ** 2 + 1),1),0)
        route_out[dir_id + 1 + len(dir)][0] = [out_left_x1_re, out_left_y1_re, out_left_x2_re, out_left_y2_re]

        for lane_id in range(1, reverse_lane_num +1):
            if direction <= 2:
                linex2_re = max(round(out_left_x2_re + 3.5 / math.sqrt(fit_stop[0] ** 2 + 1),1),0)
                liney2_re = max(round(out_left_y2_re + 3.5 * fit_stop[0] / math.sqrt(fit_stop[0] ** 2 + 1),1),0)
                linex1_re = max(round(out_left_x1_re + move_dis / math.sqrt(fit_stop[0] ** 2 + 1),1),0)
                liney1_re = max(round(out_left_y1_re + 3.5 * fit_stop[0]/ math.sqrt(fit_stop[0] ** 2 + 1),1),0)
            else:
                linex2_re = max(round(out_left_x2_re - 3.5 / math.sqrt(fit_stop[0] ** 2 + 1),1),0)
                liney2_re = max(round(out_left_y2_re - 3.5 * fit_stop[0] / math.sqrt(fit_stop[0] ** 2 + 1),1),0)
                linex1_re = max(round(out_left_x1_re - 3.5 / math.sqrt(fit_stop[0] ** 2 + 1),1),0)
                liney1_re = max(round(out_left_y1_re - 3.5 * fit_stop[0]/ math.sqrt(fit_stop[0] ** 2 + 1),1),0)
            write_lanes(child2, dir_id+len(dir), lane_id, nodes, direction_reverse_write, 0,0) # 出口道全虚线
            route_out[dir_id+1+len(dir)][lane_id] = [linex1_re,liney1_re,linex2_re,liney2_re]


    child3 = ET.SubElement(root, 'connectors')
    connector_id = 0
    for dir_id in range(len(dir)):
        route_in = dir_id + 1
        direction = dir[dir_id]['direction']
        for lane_id in range(1,len(dir[dir_id]) - 6):
            rules_in_lane = rules[dir_id][lane_id]
            connector_start_left = dir[dir_id]['Line' + str(lane_id-1)]['L1']['points'][0]
            connector_start_left = [round(start_left*scale,1) for start_left in connector_start_left]
            connector_start_right = dir[dir_id]['Line' + str(lane_id)]['L1']['points'][0]
            connector_start_right = [round(start_right * scale, 1) for start_right in connector_start_right]
            for rule in rules_in_lane:
                if rule == 'left':   # 认为左转后进入最左侧车道
                    if direction > 1:
                        out_direction = direction - 1
                        if out_direction in direction2roadNo:
                            out_route = direction2roadNo[out_direction] + len(dir)
                    else:
                        out_direction = 4
                        if out_direction in direction2roadNo:
                            out_route = direction2roadNo[4] + len(dir)
                    out_lane = 1
                    connector_end_left = [route_out[out_route][out_lane-1][0],route_out[out_route][out_lane-1][1]]
                    connector_end_right = [route_out[out_route][out_lane][0],route_out[out_route][out_lane][1]]
                    points_con = [connector_start_left,connector_start_right,connector_end_left,connector_end_right]
                    connector_id = connector_id + 1
                    write_connectors(child3, route_in, lane_id, out_route, out_lane,connector_id,points_con)
                elif rule == 'straight':  # 认为直行后进入任意车道
                    if direction <=2:
                        out_direction = direction + 2
                        if out_direction in direction2roadNo:
                            out_route = direction2roadNo[out_direction] + len(dir)
                    else:
                        out_direction = direction - 2
                        if out_direction in direction2roadNo:
                            out_route = direction2roadNo[out_direction] + len(dir)
                    if out_direction in direction2roadNo:
                        choice = out_num[out_direction]
                        for out_lane in range(1,choice+1):
                            connector_end_left = [route_out[out_route][out_lane-1][0], route_out[out_route][out_lane-1][1]]
                            connector_end_right = [route_out[out_route][out_lane][0], route_out[out_route][out_lane][1]]
                            points_con = [connector_start_left, connector_start_right, connector_end_left,connector_end_right]
                            connector_id = connector_id + 1
                            write_connectors(child3, route_in, lane_id, out_route, out_lane,connector_id,points_con)
                elif rule == 'right':  # 认为右转后进入最右侧车道
                    if direction < 4:
                        out_direction = direction + 1
                        if out_direction in direction2roadNo:
                            out_route = direction2roadNo[out_direction] + len(dir)
                    else:
                        out_direction = 1
                        if out_direction in direction2roadNo:
                            out_route = direction2roadNo[1] + len(dir)
                    if out_direction in direction2roadNo:
                        out_lane = out_num[out_direction]
                        connector_end_left = [route_out[out_route][out_lane - 1][0], route_out[out_route][out_lane - 1][1]]
                        connector_end_right = [route_out[out_route][out_lane][0], route_out[out_route][out_lane][1]]
                        points_con = [connector_start_left, connector_start_right, connector_end_left, connector_end_right]
                        connector_id = connector_id + 1
                        write_connectors(child3, route_in, lane_id, out_route, out_lane, connector_id,points_con)

    child4 = ET.SubElement(root, 'nodes')
    write_nodes(child4)



    # 将元素写入文件
    tree = ET.ElementTree(root)
    tree.write(outputpath, encoding='utf-8', xml_declaration=True)  # 写入文件并添加XML声明



if __name__ == '__main__':
    write_roads()
    # read()

