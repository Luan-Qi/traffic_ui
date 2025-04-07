import math
import xml.etree.ElementTree as ET

import numpy as np


def write_routes(child,dir_id,lane_num,length,points,mode):
    route = ET.SubElement(child, 'route')
    route.set('roadNo', str(dir_id+1))
    route.set('capacity', "0")
    route.set('name', "___")
    route.set('desp', "___")
    route.set('laneNum', str(lane_num))
    route.set('length', str(length))
    route.set('width', str(round(lane_num*3.5, 1)))
    route.set('roadLevel', "___")
    route.set('type', "1")
    if mode == 'in':
        route.set('upNodeId', "0")  # 上游路口 ，进口道为0，出口道为1
        route.set('downNodeId', "1")  # 下游路口 ，进口道为1，出口道为0
    elif mode == 'out':
        route.set('upNodeId', "1")  # 上游路口 ，进口道为0，出口道为1
        route.set('downNodeId', "0")  # 下游路口 ，进口道为1，出口道为0
    else:
        route.set('upNodeId', "0")  # 上游路口 ，进口道为0，出口道为1
        route.set('downNodeId', "0")  # 下游路口 ，进口道为1，出口道为0
    route.set('typeId', "1")
    route.set('priority', "0")
    route.set('geom', points)  #route 按行车方向两端中点坐标（x1,y1）(x2,y2)  (该坐标系与像素坐标类似，单位为m)
    route.set('spreadType', "right")
    route.set('endOffset', "0.0")
    route.set('distance', "0.0")
    route.set('createTime', "2023-05-11")
    route.set('updateTime', "2023-05-11")
    route.set('bikeLaneWidth', "0.0")

    if mode == 'in':
        return [str(0), str(1)]
    elif mode == 'out':
        return [str(1), str(0)]
    else:
        return [str(0), str(0)]


def write_lanes(child,dir_id,lane_id,nodes,direction,left,right):
    route = ET.SubElement(child, 'lane')
    route.set('roadsegId', str(dir_id+1)) # route roadNo
    route.set('laneNo', str(lane_id))  # 行驶方向从内到外依次编号
    route.set('adjacentNodeId', nodes[0])  # route upNodeId
    route.set('detectorId', "0")
    route.set('direction', direction) # 向北：1，向南：3，向东：2，向西：4
    route.set('nodeId', nodes[1])  # route downNodeId
    route.set('isLocateLamp', "false")
    route.set('isMixedLane', "false")
    route.set('isRightSpecial', "false")
    route.set('acceleration', "false")
    route.set('endOffset', "0.0")
    route.set('speed', "30.0")  # 限速
    route.set('width', "3.5")
    route.set('leftRealLineLen', str(left))
    route.set('rightRealLineLen', str(right))

def write_connectors(child,route_in,lane_in,route_out,lane_out,id,points):
    route = ET.SubElement(child, 'connector')
    route.set('nodeId', "1")  # 路口id ，画的就是观测该路口的信息，所以全部为1
    route.set('num', "0")
    route.set('contpos', "0.0")
    route.set('fromEdge', str(route_in))  # 该连接器驶入的route roadNo
    route.set('fromLane', str(lane_in))  # 该连接器驶入的lane laneNo  （roadsegId 为 route roadNo）
    route.set('indirect', "false")
    route.set('keepClear', "false")
    route.set('length', str(0))  # 连接器是贝塞尔曲线，起点终点和转折点能确定下来，程序获取长度
    route.set('isLeft', "false")
    route.set('leftlength', "0.0")
    route.set('pass',"false")
    route.set('speed', "30.0")
    route.set('toEdge', str(route_out))  # 该连接器驶离的route roadNo
    route.set('toLane', str(lane_out))   # 该连接器驶离的lane laneNo  （roadsegId 为 route roadNo）
    route.set('uncontrolled', "false")
    route.set('visibility', "0.0")
    route.set('linkIndex', str(id))  # 依次编号即可
    points = [str(point) for point in points]
    points_write = ''.join(points)
    route.set('shape', points_write)  # 连接器四个顶点的坐标

def write_nodes(child):
    route = ET.SubElement(child, 'node')
    route.set('num', "1")  # connector nodeId ，该路口的编号
    route.set('directionNum', "4")  # 进口道或出口道数
    route.set('latitude', "0.0")
    route.set('longitude', "0.0")
    route.set('keepClear', "true")
    route.set('radius', "0.0")
    route.set('tl', "0")
    route.set('x', "0.0")
    route.set('y', "0.0")
    route.set('z', "0.0")
    route.set('createTime', "0000-00-00")
    route.set('updateTime', "0000-00-00")
    route.set('fringe', "default")
    route.set('rightOfWay', "default")
    route.set('tlLayout', "opposites")
    route.set('tlType', "static")
    route.set('type', "priority")


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

    # 将元素写入文件
    tree = ET.ElementTree(root)
    tree.write(outputpath, encoding='utf-8', xml_declaration=True)  # 写入文件并添加XML声明

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
        for key in dir[dir_id]:
            if key.startswith("Line"):
                lane_id = int(key.replace("Line", ""))
                if(lane_id == 0):
                    continue
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

