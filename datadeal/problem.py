import random
import numpy as np
from datadeal.order import Order
from datadeal.orderAndDriver import Driver
from setting import *
import os
import pickle
np.random.seed(seed)


class ProblemInstance:
    def __init__(self, orderPath, driverCount):
        # load orders
        self.waitOrder = []
        if os.path.exists("tmpData/%d_%d.pl" % (month, day)):
            with open("tmpData/%d_%d.pl" % (month, day), "rb") as f:
                self.waitOrder = pickle.load(f)
        else:
            with open(orderPath, "r", encoding="utf8") as f:
                content = f.readline()
                orderCount = 0
                while content != "":
                    order = Order(content, orderCount)
                    if order.judgeLocation(XREGION + YREGION):
                        self.waitOrder.append(order)
                    content = f.readline()
                    orderCount = orderCount + 1
            if not os.path.exists("tmpData"):
                os.makedirs("tmpData")
            with open("tmpData/%d_%d.pl" % (month, day), "wb") as f:
                pickle.dump(self.waitOrder, f)

        # self.waitOrder.sort(key=lambda x: x.pickTime)
        self.startTime = self.waitOrder[0].pickTime
        self.endTime = self.startTime + 60 * 60 * 24
        # load drivers
        select = np.random.randint(len(self.waitOrder), size=driverCount)
        self.drivers = [Driver(self.waitOrder[i].getPickLocation()) for i in select]

    def batch(self, currentTimestamp):
        ords = iter(self.waitOrder)
        single_order = next(ords)
        orders = []
        while single_order.pickTime < currentTimestamp:
            # if single_order.pickTime + single_order.durable <= currentTimestamp or not single_order.available:
            #     self.waitOrder.remove(single_order)
            # else:
            #     orders.append(single_order)
            # single_order = next(ords)

            single_order.maxWait = base_wait_time + random.randint(0, wait_time_noise)
            orders.append(single_order)
            try:
                single_order = next(ords)
            except Exception as e:
                break
        del(self.waitOrder[0: self.waitOrder.index(single_order)])
        drivers = list(filter(lambda x: x.relaxTime < currentTimestamp, self.drivers))
        # curIdMap = [None for x in range(len(orders))]
        # for i in range(len(orders)):
        #     curIdMap[i] = orders[i].id
        return orders, drivers


if __name__ == '__main__':
    problemInstance = ProblemInstance(data_path, 1000)
    print()


