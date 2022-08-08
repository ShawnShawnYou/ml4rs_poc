import sys
sys.path.append("..")

import numpy

from experiment.gas import in_which_grid
from setting import *
from datadeal.problem import ProblemInstance
from experiment.solve import solve
from experiment.costSaving import get_original_id_by_mapped, cost_saving, transfer_id_map, sort_total_cost
import time
from tqdm import tqdm



def get_data(now_order, partner, rate, start_time, heatmap, flag):
    # now_order = reformat_orders[order_id]
    # partner = reformat_orders[max_cost_saving_partner[order_id]]

    order_start_grid = in_which_grid(now_order.pickX, now_order.pickY)
    order_end_grid = in_which_grid(now_order.dropX, now_order.dropY)

    partner_start_grid = in_which_grid(partner.pickX, partner.pickY)
    partner_end_grid = in_which_grid(partner.dropX, partner.dropY)

    data = [order_start_grid[0],
            order_start_grid[1],
            order_end_grid[0],
            order_end_grid[1],
            partner_start_grid[0],
            partner_start_grid[1],
            partner_end_grid[0],
            partner_end_grid[1],

            heatmap[order_start_grid[0]][order_start_grid[1]],
            heatmap[order_end_grid[0]][order_end_grid[1]],
            heatmap[partner_start_grid[0]][partner_start_grid[1]],
            heatmap[partner_end_grid[0]][partner_end_grid[1]],

            int(rate / 5),  # shareability

            int((int(now_order.pickTime) - start_time) / 60),   # 订单发布时间
            int(max((int(partner.pickTime) - int(now_order.pickTime)) / 60, 0)),    # 订单等待时间，如果匹配到之前的订单就是0

            flag
            ]

    return data


def get_heatmap(orders):
    ret = [[0 for y in range(NUM_GRID_Y)] for x in range(NUM_GRID_X)]
    for order in orders:
        order_start_grid = in_which_grid(order.pickX, order.pickY)
        order_end_grid = in_which_grid(order.dropX, order.dropY)
        ret[order_start_grid[0]][order_start_grid[1]] += 1
        ret[order_end_grid[0]][order_end_grid[1]] += 1
    return ret


def grab():
    problem = ProblemInstance(data_path, 100000)

    dataset = []

    last_round_order = []
    current_time = problem.startTime + fragment

    max_cost_saving = {}
    max_cost_saving_partner = {}
    update_flag = {}
    reformat_orders = {}

    for now_round in tqdm(range(total_round)):
        if not (problem.waitOrder and current_time < problem.endTime):
            break

        # 获取数据
        orders, drivers = problem.batch(current_time)

        for order in orders:
            reformat_orders[order.id] = order

        orders += last_round_order
        heatmap = get_heatmap(orders)

        # 算法运行

        t = cost_saving(orders)
        for order_id in t.keys():
            t[order_id] = sorted(t[order_id], key=lambda myClass: myClass.rate, reverse=True)
            if order_id not in max_cost_saving.keys():
                max_cost_saving[order_id] = t[order_id][0].rate
                max_cost_saving_partner[order_id] = t[order_id][0].match_id
                update_flag[order_id] = 0

            if max_cost_saving[order_id] < t[order_id][0].rate:

                data = get_data(reformat_orders[order_id], reformat_orders[max_cost_saving_partner[order_id]],
                                t[order_id][0].rate, problem.startTime, heatmap, 1)

                dataset.append(data)
                max_cost_saving[order_id] = t[order_id][0].rate
                max_cost_saving_partner[order_id] = t[order_id][0].match_id
                update_flag[order_id] = 1

        # dataset.sort(key=(lambda x:x[0]))
        current_time += fragment
        last_round_order = []

        for order_id in t.keys():
            order = reformat_orders[order_id]
            if order.pickTime + w * window_size > current_time:     # 还可以等
                last_round_order.append(order)
            elif update_flag[order.id] == 0:
                data = get_data(reformat_orders[order.id], reformat_orders[max_cost_saving_partner[order.id]],
                                t[order.id][0].rate, problem.startTime, heatmap, 0)
                dataset.append(data)




    # numpy 清洗 + 存储
    numpy.save('new_data_%d.%d_%d' % (month, day, w), dataset)

    print(len(dataset))



def data_filter():
    # 这个函数没啥用，用来测试的
    original_data = numpy.load('original_data.npy', allow_pickle=True)


    problem = ProblemInstance(data_path, 100000)
    last_round_order = []
    current_time = problem.startTime + fragment


    reformat_orders = {}

    return




if __name__ == '__main__':
    grab()
    # data_filter()