from experiment.costSaving import *
from setting import *
from blossom_match import blossom_maximum_match


def in_which_grid(x, y,
                  num_x=NUM_GRID_X, num_y=NUM_GRID_Y, x_region=XREGION, y_region=YREGION):
    single_x = (x_region[1] - x_region[0]) / num_x
    single_y = (y_region[1] - y_region[0]) / num_y
    if x_region[0] < x < x_region[1] and y_region[0] < y < y_region[1]:
        return int((x - x_region[0]) / single_x), int((y - y_region[0]) / single_y)
    return -1, -1


def group_order_by_gird(orders):

    order_group_list = [[] for i in range(NUM_GRID_X * NUM_GRID_Y)]

    for order in orders:
        x_index, y_index = in_which_grid(order.pickX, order.pickY)

        if (x_index, y_index) == (-1, -1):
            continue
        else:
            order_group_list[x_index + y_index].append(order)

    # return order_group_list
    return [orders]


def in_edge_list(u, v, edge_list):
    for i in edge_list:
        if i[0] == u and i[1] == v:
            return True
        if i[0] == v and i[1] == u:
            return True

    return False


def gas_match(orders):

    measurement = {
        'size': 0,
        'unmatched': 0,
        'matched': 0,

        'total_cost_saving': 0,
        'running_time': 0
    }

    order_group_list = group_order_by_gird(orders)

    for order_list in order_group_list:
        t = cost_saving(order_list)

        transfer_t, original_individual_cost_saving, original_total_cost_saving, id_map = transfer_id_map(t)

        edge_list = []

        for i in range(len(transfer_t)):
            for j in range(len(transfer_t[i])):
                total_cost_saving = original_total_cost_saving[i][j]
                "************修正individual cost saving**********"
                if total_cost_saving < 0:
                    total_cost_saving = 0
                "***********************************************"
                total_cost_saving = round(total_cost_saving, 6)
                total_cost_saving *= 1000000
                total_cost_saving = int(total_cost_saving)


                tmp = [i, transfer_t[i][j] - 1, total_cost_saving]
                if not in_edge_list(i, transfer_t[i][j] - 1, edge_list):
                    edge_list.append(tmp)

        total_cost_saving, match = blossom_maximum_match(len(transfer_t), edge_list)
        total_cost_saving /= 1000000
        # print(total_cost_saving)
        # print(match)

        for i in range(len(match)):
            if match[i] is not None:
                measurement['matched'] += 1
            else:
                measurement['unmatched'] += 1
        measurement['size'] += len(match)
        measurement['total_cost_saving'] += total_cost_saving



    return measurement, match



if __name__ == '__main__':
    # for test
    from datadeal.problem import ProblemInstance

    problem = ProblemInstance(data_path, 100000)
    current_time = problem.startTime + 120
    orders, drivers = problem.batch(current_time)

    measurement = gas_match(orders)
    print(measurement)



