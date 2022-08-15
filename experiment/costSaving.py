from copy import deepcopy
from typing import List
from datadeal.order import Order
from setting import *
from tqdm import tqdm
import math
import sys

sys.path.append(algorithm_path)
from BFRM import basic_stable_roommate_matching
from GFRM import greedy_roommate_matching
from algorithm_config import set_config


def ManhaPick2Pick(a: Order, b: Order):
    return abs(a.pickX-b.pickX)+abs(a.pickY-b.pickY)


def ManhaPick2Drop(a: Order, b: Order):
    return abs(a.pickX-b.dropX)+abs(a.pickY-b.dropY)


def ManhaDrop2Drop(a: Order, b: Order):
    return abs(a.dropX-b.dropX)+abs(a.dropY-b.dropY)


class myClass:
    def __init__(self, id, match_id, save_total, save_individual, rate):
        self.id = id
        self.match_id = match_id
        self.save_total = save_total
        self.save_individual = save_individual
        self.rate = rate

    def __repr__(self):
        return str(self.id)+' '+str(self.match_id)+' '+str(self.save_total)+' '+str(self.save_individual)+'\n'


def cost_saving(orders: List[Order]):
    ptable = {}

    clear_orders = []
    for i in range(len(orders)):
        if not (orders[i].dropX - orders[i].pickX == 0 and
                orders[i].dropY - orders[i].pickY == 0):
            clear_orders.append(orders[i])
        else:
            to_be_delete = orders[i]

    del orders
    orders = clear_orders


    for i in range(len(orders)):
        key = orders[i].id
        plist = []
        for j in range(len(orders)):
            if i == j:
                continue
            vix = orders[i].dropX - orders[i].pickX
            vjx = orders[j].dropX - orders[j].pickX

            viy = orders[i].dropY - orders[i].pickY
            vjy = orders[j].dropY - orders[j].pickY

            cosTheta = (vix * vjx + viy * vjy) / (math.sqrt(vix * vix + viy * viy) * math.sqrt(vjx * vjx + vjy * vjy))

            if cosTheta > 1:
                cosTheta = 1
            if cosTheta < -1:
                cosTheta = -1

            Theta=math.acos(cosTheta)*57.3 # 1弧度约等于57.3度
            if Theta < 90 and ManhaPick2Pick(orders[i], orders[j])/orders[i].speed < orders[i].maxWait and \
                    ManhaPick2Pick(orders[i],orders[j])/orders[j].speed < orders[j].maxWait:
                com = ManhaPick2Pick(orders[i], orders[j])+ManhaDrop2Drop(orders[i], orders[j])
                plan1 = com+ManhaPick2Drop(orders[j],orders[i])
                plan2 = com+ManhaPick2Drop(orders[j],orders[j])
                plan3 = com+ManhaPick2Drop(orders[i],orders[i])
                plan4 = com+ManhaPick2Drop(orders[i],orders[j])
                share = min(plan1, plan2, plan3, plan4)
                if share == plan1:
                    ci = ManhaPick2Pick(orders[i],orders[j])+ManhaPick2Drop(orders[j],orders[i])/2
                    # cj = ManhaPick2Drop(orders[j],orders[i])/2+ManhaDrop2Drop(orders[i],orders[j])
                if share == plan2:
                    ci = com +ManhaPick2Drop(orders[j],orders[j])/2
                    # cj = ManhaPick2Drop(orders[j],orders[j])/2
                if share == plan3:
                    ci = ManhaPick2Drop(orders[i],orders[i])/2
                    # cj = com +ManhaPick2Drop(orders[i],orders[i])/2
                if share == plan4:
                    ci = ManhaPick2Drop(orders[i],orders[j])/2+ManhaDrop2Drop(orders[j],orders[i])
                    # cj = ManhaPick2Pick(orders[j],orders[i])+ManhaPick2Drop(orders[i],orders[j])/2
                d_sum = orders[i].absluteDistance + orders[j].absluteDistance

                rate = orders[i].absluteDistance / d_sum if d_sum != 0 else 0

                if d_sum == 0:
                    save_total = 0
                    save_individual = 0
                else:
                    save_total = d_sum - share
                    save_individual = orders[i].absluteDistance-ci
                save_individual /= orders[i].absluteDistance
                save_individual_j = (orders[j].absluteDistance - (share - ci)) / orders[j].absluteDistance
                    # if save_total < 0:
                    #     save_total = 0
                    # if save_individual < 0:
                    #     save_individual = 0


                ptable.setdefault(key, plist).append(
                    myClass(orders[i].id, orders[j].id,
                            int(save_total * 10**8), save_individual,
                            round(rate * 100, 2)))
    return ptable


def transfer_id_map(t):
    t = list(t.values())
    # print(sorted(t[len(t) - 1], key=lambda myClass: myClass.save_total))

    id_map = {}
    for i in range(len(t)):
        tmp = {t[i][0].id: i + 1}
        id_map.update(tmp)

    transfer_t = []
    t_individual_cost_saving = []
    t_total_cost_saving = []
    for i in range(len(t)):
        # transfer_t[i] = [t[i][j].match_id for j in range(len(t[i]))]
        t[i] = sorted(t[i], key=lambda myClass: myClass.save_individual, reverse=True)
        tmp = []
        tmp_individual_cost_saving = []
        tmp_total_cost_saving = []

        for j in range(len(t[i])):
            tmp.insert(j, id_map[t[i][j].match_id])
            tmp_individual_cost_saving.insert(j, t[i][j].save_individual)
            tmp_total_cost_saving.insert(j, t[i][j].save_total)

        transfer_t.insert(i, tmp)
        t_individual_cost_saving.insert(i, tmp_individual_cost_saving)
        t_total_cost_saving.insert(i, tmp_total_cost_saving)

    return transfer_t, t_individual_cost_saving, t_total_cost_saving, id_map


def sort_total_cost(t, id_map):
    # push 进 list 排序 转id 丢到算法里
    t = list(t.values())
    sorted_edge_list = []

    for i in range(len(t)):
        for j in range(len(t[i])):
            sorted_edge_list.append(t[i][j])

    sorted_edge_list = sorted(sorted_edge_list, key=lambda myClass: myClass.save_total, reverse=True)

    for i in range(len(sorted_edge_list)):
        sorted_edge_list[i].id = id_map[sorted_edge_list[i].id]
        sorted_edge_list[i].match_id = id_map[sorted_edge_list[i].match_id]

    return sorted_edge_list


def get_original_id_by_mapped(mapped_id, id_map, start_with=1):
    for key in id_map:
        if id_map[key] == mapped_id + start_with:
            return key
    return None


def test():
    from datadeal.problem import ProblemInstance

    batch_gap = 60
    problemInstance = ProblemInstance(data_path, 1000)
    currentTime = problemInstance.startTime + batch_gap

    count_list = [0 for i in range(5)]
    empty = 0
    avg_count_len = 0

    debug = False
    count = 1000
    total_different_count = 0
    for round in tqdm(range(count)):

        # 以下是准备阶段
        orders, drivers = problemInstance.batch(currentTime)
        currentTime = currentTime + batch_gap
        t = cost_saving(orders)

        transfer_t, original_individual_cost_saving, original_total_cost_saving, id_map = transfer_id_map(t)
        sorted_edge_list = sort_total_cost(t, id_map)

        # 以下是统计和算法阶段
        first_pair_count = 0
        for i in range(len(transfer_t)):
            first = transfer_t[i][0] - 1
            if transfer_t[first][0] == i + 1:
                first_pair_count = first_pair_count + 1

        test_data = transfer_t

        count_len = 0
        for i in range(len(test_data)):
            count_len = count_len + len(test_data[i])
        count_len = count_len / len(test_data)
        avg_count_len = avg_count_len + count_len

        if debug:
            print("preferences", test_data)

        # 算法主体
        set_config('strategy', 2)
        test_data_2 = deepcopy(test_data)
        result_type, msg, result = basic_stable_roommate_matching(test_data, original_individual_cost_saving)
        match = greedy_roommate_matching(test_data_2, sorted_edge_list, start_with=1)

        different_count = 0
        single_count = 0
        for i in range(len(result)):
            if result[i] != match[i]:
                # print(i, ":", "BFRM:", result[i], "GFRM:", match[i])
                different_count = different_count + 1
            if len(match[i]) != 1:
                single_count = single_count + 1
        # print(different_count)
        # print(single_count)
        total_different_count = total_different_count + different_count

        # print(result)
        # print(match)

        for i in result:
            if len(i) == 0:
                empty = empty + 1

        count_list[result_type] = count_list[result_type] + 1

        if debug:
            print(msg, result)
            print()

    summary = {'avg_different_count': total_different_count / count,
               'avg_count_len': avg_count_len / count,
               'avg empty': empty / count,
               'bug': count_list[0],
               'after phase_1 find answer': count_list[1],
               'unsolvable by phase_1': count_list[2],
               'after phase_2 find answer': count_list[3],
               'unsolvable by phase_2': count_list[4]}
    print(summary)


if __name__ == '__main__':
    test()
