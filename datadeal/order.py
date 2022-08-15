import time
import numpy as np
from setting import *
np.random.seed(seed)


class Order:
    def __init__(self, dataLine, id=-1):
        matched = 0
        match_id = None
        nextBatchedOrder = []
        self.id = id
        dataList = dataLine.strip().split(',')
        self.pickTime = time.mktime(time.strptime(dataList[1], '%Y-%m-%d %H:%M:%S'))
        self.dropoffTime = time.mktime(time.strptime(dataList[2], '%Y-%m-%d %H:%M:%S'))
        self.passengerCount = int(dataList[3])
        self.tripDistance = float(dataList[4])
        self.maxWait = int(dataList[-1])
        self.pickX = float(dataList[5])
        self.pickY = float(dataList[6])
        try:
            self.dropX = float(dataList[9])
            self.dropY = float(dataList[10])
        except:
            self.dropY = self.pickY
            self.dropX = self.pickX
        self.durable = np.random.randint(200) + 200
        self.absluteDistance = abs(self.pickX - self.dropX) + abs(self.pickY - self.dropY)

        if (self.dropoffTime - self.pickTime) == 0:
            self.speed = 0.00009 #每秒移动的经纬度
        else:
            self.speed = self.absluteDistance / (self.dropoffTime - self.pickTime)
            if self.speed > 0.00015:
                self.speed = 0.00015
            if self.speed < 0.00006:
                self.speed = 0.00006
        self.deadline = (self.absluteDistance / self.speed) * 1.5 + self.durable
        self.totalAmount = float(dataList[-2])
        self.available = True

    def __add__(self, order):
        self.deadline = min(self.deadline, order.deadline)
        self.durable = min(self.durable, order.durable)
        self.totalAmount = self.totalAmount + order.totalAmount
        self.speed = (self.absluteDistance + order.absluteDistance) / (self.dropoffTime + order.dropoffTime - self.pickTime - order.pickTime)


    def judgeLocation(self, range_xy):
        if self.pickX == self.dropX and self.pickY == self.dropY:
            return False
        if range_xy[0] < self.pickX < range_xy[1] and range_xy[2] < self.pickY < range_xy[3] and range_xy[
            0] < self.dropX < range_xy[1] and range_xy[2] < self.dropY < range_xy[3]:
            return True
        return False

    def getPickLocation(self):
        return self.pickX, self.pickY

    def getDropLocation(self):
        return self.dropX, self.dropY

    def run_out_of_time(self, current_time):
        if self.pickTime + self.maxWait < current_time + fragment:
            return False
        return True

    def groupOrder(self, match_order_id, currentTime):
        if match_order_id: # 若match_order不为空，即self在这个batch得到了匹配对象
            matched = 1
            return matched
        # 若match_order为空，无匹配对象，判断order若丢到下个batch是否会超时
        elif self.pickTime + self.maxWait < currentTime + fragment:
            matched = -1
            self.nextBatchedOrder.append(self) # 如何将这些order传入problem.py文件中的batch函数中的orders当中
        # 没有超时的话，matched保持为0
        return matched

    def toString(self, id, pickRegin, dropRegin):
        return ",".join([str(id), str(pickRegin), str(dropRegin), str(self.pickX), str(self.pickY), str(self.dropX),
                         str(self.dropY),
                         str(self.pickTime), str(self.dropoffTime),
                         str(self.totalAmount), str(self.tripDistance), str(np.random.randint(10) + 4)])