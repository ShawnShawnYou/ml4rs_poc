from experiment.costSaving import *

sys.path.append(algorithm_path)
from BFRM import basic_stable_roommate_matching
from GFRM import greedy_roommate_matching
from algorithm_config import set_config


def get_order(target_id, order_list):
    for order in order_list:
        if order.id == target_id:
            return order
    return None


def solve(orders, current_time, last_round_orders,
          algorithm=1, with_G=True):

    # 预处理
    t = cost_saving(orders)
    transfer_t, original_individual_cost_saving, original_total_cost_saving, id_map = transfer_id_map(t)
    sorted_edge_list = sort_total_cost(t, id_map)

    test_data = deepcopy(transfer_t)
    test_data_2 = deepcopy(transfer_t)

    # 算法
    result_type = None
    if algorithm != 0:
        set_config('strategy', algorithm)
        result_type, msg, result = basic_stable_roommate_matching(test_data, original_individual_cost_saving)
        match = result
    else:
        if with_G:
            match = greedy_roommate_matching(test_data_2, sorted_edge_list, start_with=1)
        else:
            raise Exception('bug on algorithm setting: strategy=0, with_G=False')

    # GFRM清洗
    order_list_GFRM = []
    index_list_GFRM = []
    if algorithm != 0 and with_G and (result_type == 2 or result_type == 4):

        for i in range(len(match)):
            if len(match[i]) != 1 or match[i][0] == i:
                original_index = get_original_id_by_mapped(i, id_map)
                order = get_order(original_index, orders)

                order_list_GFRM.append(order)
                index_list_GFRM.append(i)
                # index_list_GFRM 里放的是在match里面的index

        t_GFRM = cost_saving(order_list_GFRM)
        transfer_t_GFRM, original_individual_cost_saving_GFRM, original_total_cost_saving_GFRM, id_map_GFRM = transfer_id_map(t_GFRM)
        sorted_edge_list_GFRM = sort_total_cost(t_GFRM, id_map_GFRM)

        match_GFRM = greedy_roommate_matching(transfer_t_GFRM, sorted_edge_list_GFRM, start_with=1)

        for i in range(len(match_GFRM)):
            original_id = get_original_id_by_mapped(i, id_map_GFRM)
            # 订单的id
            index_match = id_map[original_id] - 1

            match[index_match] = []
            for j in range(len(match_GFRM[i])):
                original_id_j = get_original_id_by_mapped(match_GFRM[i][j], id_map_GFRM)

                index_match_j = id_map[original_id_j] - 1

                match[index_match].insert(j, index_match_j)

    # 过滤结果
    for i in range(len(match)):
        original_id = get_original_id_by_mapped(i, id_map)
        order = get_order(original_id, orders)

        if len(match[i]) == 1 and match[i][0] != i:
            # 有解
            order.match_id = get_original_id_by_mapped(match[i][0], id_map)
        else:
            # 无解
            if order.run_out_of_time(current_time):
                order.match_id = order.id
                match[i] = [i]
            else:
                last_round_orders.append(order)
                match[i] = []

    return match, t, transfer_t, id_map


if __name__ == '__main__':
    # for test
    from datadeal.problem import ProblemInstance

    problemInstance = ProblemInstance(data_path, 1000)
    solve(problemInstance, 1, True)