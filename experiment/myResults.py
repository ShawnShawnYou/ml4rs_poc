import random
import time
from copy import deepcopy

from tqdm import tqdm

from experiment.gas import gas_match
from setting import *
from datadeal.problem import ProblemInstance
from experiment.solve import solve
from experiment.costSaving import get_original_id_by_mapped, cost_saving, transfer_id_map
import sys

sys.path.append(algorithm_path)
from preference_util import verify_match, fairness_compute


def experiment(total_round=3000, algorithm_strategy=2, with_G_strategy=True):

    problem = ProblemInstance(data_path, 100000)

    # 实验参数
    skip_bug = True

    # 实验指标
    overall_measurement = {
        'size': 0,
        'unmatched': 0,
        'matched': 0,
        'match_rate': 0,
        'unfair': 0,

        'total_cost_saving': 0,
        'running_time': 0
    }

    # 实验开始
    start_time = time.time()
    current_time = problem.startTime + fragment
    last_round_orders = []
    for now_round in range(total_round):
        if not (problem.waitOrder and current_time < problem.endTime):
            break
        # 只要还有order即乘客没有被服务，或当前时间还在endTime之前，endTime是第一个order的时间加了一天时间得到的（即我们只跑1 day的数据）
        # 跑的是一天的数据，但是一个batch一个batch的运行

        # 算法指标
        measurement = {
            'size': 0,
            'unmatched': 0,
            'matched': 0,
            'match_rate': 0,
            'unfair': 0,

            'total_cost_saving': 0,
            'running_time': 0
        }

        # 获取数据
        orders, drivers = problem.batch(current_time)
        orders += last_round_orders
        last_round_orders = []

        # 算法运行
        match, t, transfer_t, id_map = solve(orders=orders, current_time=current_time, last_round_orders=last_round_orders,
                                             algorithm=algorithm_strategy, with_G=with_G_strategy)

        # 算法正确初筛（匹配错误）
        if len(match) == 0:
            continue
        match, verification_result = verify_match(match, skip_bug)

        unfair_count, none_count = fairness_compute(transfer_t, match)

        for i in range(len(match)):
            if len(match[i]) == 0:
                continue
            partner = match[i][0]
            if partner == i:
                continue
            order_id_i = get_original_id_by_mapped(i, id_map)

            my_class_list_i = t[order_id_i]

            find_flag = False
            for j in my_class_list_i:
                if j.match_id == partner + 1:
                    "**************修正save_individual*************"
                    if j.save_individual < 0:
                        j.save_individual = 0
                    "***************************"
                    measurement['total_cost_saving'] = measurement['total_cost_saving'] + j.save_individual
                    find_flag = True

            if not find_flag:
                raise Exception("bug")

        measurement['size'] = len(match)
        measurement['unmatched'] = verification_result['single_dog'] + verification_result['wrong_match'] + verification_result['long_list']
        measurement['matched'] = verification_result['has_partner']
        measurement['unfair'] = unfair_count

        overall_measurement['size'] += measurement['size']
        overall_measurement['matched'] += measurement['matched']
        overall_measurement['unmatched'] += measurement['unmatched']
        overall_measurement['total_cost_saving'] += measurement['total_cost_saving']
        overall_measurement['unfair'] += measurement['unfair']

        # 收尾
        current_time = current_time + fragment
    end_time = time.time()

    overall_measurement['match_rate'] = overall_measurement['matched'] / overall_measurement['size']
    overall_measurement['running_time'] = end_time - start_time
    overall_measurement['total_cost_saving'] /= overall_measurement['size']

    # d = datetime.now()
    # with open("result/%s_%d_%d_%d_%s.txt" % (algorithm, month, day, fragment, d.strftime("%d-%H-%M")),
    #           "w", encoding="utf8") as f:
    #     f.write(str(overall_measurement))

    return overall_measurement


def experiment_2(total_round=1000, algorithms=[1, 3]):

    problem = ProblemInstance(data_path, 100000)

    skip_bug = True

    # 实验指标
    overall_measurement = [None, None]
    for i in range(len(overall_measurement)):
        overall_measurement[i] = {
            'size': 0,
            'total_cost_saving': 0,
        }

    # 实验开始
    current_time = problem.startTime + fragment
    last_round_orders = []
    for now_round in range(total_round):
        if not (problem.waitOrder and current_time < problem.endTime):
            break
        # 只要还有order即乘客没有被服务，或当前时间还在endTime之前，endTime是第一个order的时间加了一天时间得到的（即我们只跑1 day的数据）
        # 跑的是一天的数据，但是一个batch一个batch的运行

        # 算法指标
        measurement = [None, None]
        for i in range(len(measurement)):
            measurement[i] = {
                'size': 0,
                'total_cost_saving': 0,
            }

        # 获取数据
        orders, drivers = problem.batch(current_time)

        orders_list = [deepcopy(orders), deepcopy(orders)]

        # 算法运行
        match_compare = [None, None]

        for i in range(2):
            match_compare[i], t, transfer_t, id_map = solve(orders=orders_list[i], current_time=current_time,
                                                last_round_orders=last_round_orders,
                                                algorithm=algorithms[i], with_G=False)

        match_1 = match_compare[0]
        match_3 = match_compare[1]

        match_tmp = [None, None]
        for i in range(2):
            match_tmp[i] = [[j] for j in range(len(match_1))]


        different_count = 0
        for i in range(len(match_1)):
            match_tmp[0][i] = match_1[i]
            match_tmp[1][i] = match_3[i]
            if match_1[i] != match_3[i]:
                different_count = different_count + 1

        del match_1, match_3


        for tmp_i in range(2):

            match = match_tmp[tmp_i]
            # 算法正确初筛（匹配错误）
            if len(match) == 0:
                continue
            match, verification_result = verify_match(match, skip_bug)

            for i in range(len(match)):
                if len(match[i]) == 0:
                    continue
                partner = match[i][0]
                if partner == i:
                    continue
                order_id_i = get_original_id_by_mapped(i, id_map)

                my_class_list_i = t[order_id_i]

                find_flag = False
                for j in my_class_list_i:
                    if j.match_id == partner + 1:
                        "************修正individual cost saving**********"
                        # if j.save_individual < 0:
                        #     print(j.save_individual)
                        #     j.save_individual = 0
                        "***********************************************"
                        measurement[tmp_i]['total_cost_saving'] += j.save_individual
                        find_flag = True

                if not find_flag:
                    raise Exception("bug")

            measurement[tmp_i]['size'] = different_count

            overall_measurement[tmp_i]['size'] += measurement[tmp_i]['size']
            overall_measurement[tmp_i]['total_cost_saving'] += measurement[tmp_i]['total_cost_saving']

        # 收尾
        current_time += fragment

    for i in range(2):
        overall_measurement[i]['total_cost_saving'] /= overall_measurement[i]['size']

    print(overall_measurement[0])
    print(overall_measurement[1])


def experiment_gas(total_round=1000):
    problem = ProblemInstance(data_path, 100000)

    # 实验指标
    overall_measurement = {
        'size': 0,
        'unmatched': 0,
        'matched': 0,
        'match_rate': 0,
        'unfair': 0,

        'total_cost_saving': 0,
        'running_time': 0
    }

    # 实验开始

    start_time = time.time()
    current_time = problem.startTime + fragment
    for now_round in range(total_round):
        if not (problem.waitOrder and current_time < problem.endTime):
            break

        # 获取数据
        orders, drivers = problem.batch(current_time)

        # 算法运行
        measurement, match = gas_match(orders)

        t = cost_saving(orders)
        transfer_t, original_individual_cost_saving, original_total_cost_saving, id_map = transfer_id_map(t)
        for i in range(len(match)):
            if match[i] is None:
                continue
            match[i] = [match[i]]
        unfair_count, none_count = fairness_compute(transfer_t, match)


        overall_measurement['size'] += measurement['size']
        overall_measurement['matched'] += measurement['matched']
        overall_measurement['unmatched'] += measurement['unmatched']
        overall_measurement['total_cost_saving'] += measurement['total_cost_saving']
        overall_measurement['unfair'] += (unfair_count + none_count)

        current_time += fragment
    end_time = time.time()

    overall_measurement['match_rate'] = overall_measurement['matched'] / overall_measurement['size']
    overall_measurement['running_time'] = end_time - start_time
    overall_measurement['total_cost_saving'] /= overall_measurement['size']

    return overall_measurement


if __name__ == '__main__':
    # experiment_gas()
    # # experiment 1
    # overall_measurement = experiment_gas(total_round)
    # print('algorithm_strategy:', 'gas',
    #       'with_G:', 'None',
    #       overall_measurement)
    for algorithm_strategy in [0, 1, 2, 3]:
        for with_G in [True, False]:
            if algorithm_strategy == 0 and with_G == False:
                continue
            overall_measurement = experiment(total_round, algorithm_strategy, with_G)
            print('algorithm_strategy:', algorithm_strategy,
                  'with_G:', with_G,
                  overall_measurement)


    # experiment_2()
