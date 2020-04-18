def traverse(goal_state, prev):
    '''
    extract a plan using the result of dijkstra's algorithm
    :param goal_state: the end state
    :param prev: result of dijkstra's algorithm
    :return: a list of (state, actions) such that the first element is (start_state, a_0), and the last is
    (goal_state, None)
    '''
    result = [(goal_state, None)]
    cur_state = goal_state
    nxt_state = prev[goal_state.to_string()]
    actions = nxt_state.get_actions()
    while nxt_state:
        actions = nxt_state.get_actions()
        for a in actions:
            if nxt_state.apply_action(a).is_same(cur_state):
                result.append((nxt_state, a))
                cur_state = nxt_state
                nxt_state = prev[nxt_state.to_string()]
                break

    return result


def print_plan(plan):
    print('plan length {}'.format(len(plan)-1))
    for current_state, action in plan:
        print(current_state.to_string())
        if action is not None:
            print('apply action {}'.format(action))
