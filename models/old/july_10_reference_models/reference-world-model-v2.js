// Define the gridworld map with various locations
var gridworld = [
    ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'H'],
    ['S', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'S'],
    ['S', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'S'],
    ['S', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'S'],
    ['S', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'S'],
    ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'C'],
    ['S', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'S'],
    ['S', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'S'],
    ['S', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'S'],
    ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'B']
];

// Define the list of goals in the gridworld
var goals = ['hospital', 'busStop', 'coffeeShop'];

// Initial coordinates of the agent
var initial_x = 1;
var initial_y = 3;

// Possible directions the agent can move, including diagonal directions
var directions = ['west', 'east', 'north', 'south', 'north-west', 'north-east', 'south-west', 'south-east'];

// Define utility constants for different goals
var HOSPITAL_MEAN_UTILITY = 1000;
var BUSSTOP_MEAN_UTILITY = 100;
var COFFEESHOP_MEAN_UTILITY = 10;
var HOSPITAL_UTILITY_VARIANCE = 100;
var BUSSTOP_UTILITY_VARIANCE = 10;
var COFFEESHOP_UTILITY_VARIANCE = 1;

// Define the utility of goals
var goal_utility = cache(function(agentId, goalType) {
  switch (goalType) {
    case 'hospital': return gaussian(HOSPITAL_MEAN_UTILITY, HOSPITAL_UTILITY_VARIANCE);
    case 'busStop': return gaussian(BUSSTOP_MEAN_UTILITY, BUSSTOP_UTILITY_VARIANCE);
    case 'coffeeShop': return gaussian(COFFEESHOP_MEAN_UTILITY, COFFEESHOP_UTILITY_VARIANCE);
    default: return 0;
  }
});

// Define the utility of motions based on the location type
var motion_utility = cache(function(agentId, locationType, motionType) {
  if (locationType === 'G') {
    if (motionType === 'is_walking') return -0.2;
    if (motionType === 'is_staying') return 0;
  } else {
    if (motionType === 'S') return -0.2;
    if (motionType === 'is_staying') return 0;
  }
  return 0;
});

// Define the utility based on the location type
var location_utility = cache(function(agentId, locationType) {
  switch (locationType) {
    case 'G':
    case 'C': return 0;
    default: return goal_utility(agentId, locationType);
  }
});

// Get the type of location in the gridworld at coordinates (x, y)
var get_gridworld_at = function(gridworld, x, y) {
  return gridworld[y][x];
};

// Increment x-coordinate based on direction
var x_increment = function(direction) {
  switch (direction) {
    case 'west': return -1;
    case 'east': return 1;
    case 'north':
    case 'south': return 0;
    case 'north-west': return -1;
    case 'north-east': return 1;
    case 'south-west': return -1;
    case 'south-east': return 1;
    case 'stay': return 0;
  }
};

// Increment y-coordinate based on direction
var y_increment = function(direction) {
  switch (direction) {
    case 'north': return -1;
    case 'south': return 1;
    case 'west':
    case 'east': return 0;
    case 'north-west':
    case 'north-east': return -1;
    case 'south-west':
    case 'south-east': return 1;
    case 'stay': return 0;
  }
};

// Get the maximum x-dimension of the gridworld
var gridworld_max_x = function(gridworld) {
  return gridworld[0].length;
};

// Get the maximum y-dimension of the gridworld
var gridworld_max_y = function(gridworld) {
  return gridworld.length;
};

// Transition to the next state in the gridworld based on action
var gridworld_transition = function(gridworld, current_x, current_y, action) {
  var direction = action;
  var next_x = current_x + x_increment(direction);
  if (next_x >= gridworld_max_x(gridworld) || next_x < 0) next_x = current_x;
  var next_y = current_y + y_increment(direction);
  if (next_y >= gridworld_max_y(gridworld) || next_y < 0) next_y = current_y;
  return { location: get_gridworld_at(gridworld, next_x, next_y), x: next_x, y: next_y };
};

// Define the total utility function based on the state and action
var utility_function = cache(function(agentId, gridworld, state_x, state_y, action) {
  var location_type = get_gridworld_at(gridworld, state_x, state_y);
  var state_location_utility = location_utility(agentId, location_type);
  var state_motion_utility = motion_utility(agentId, location_type, action);
  return state_location_utility + state_motion_utility;
});

// Value function to calculate the utility at a given state and iteration
var value_function = cache(function(agentId, curr_iteration, gridworld, state_x, state_y) {
  // Base case: if no iterations left, return 0
  if (curr_iteration === -1) return 0;
  // Get the value of the optimal action from the previous iteration
  var prev_optimal_action_value = optimal_action_value(agentId, curr_iteration - 1, gridworld, state_x, state_y);
  // Return the value of the optimal action
  return prev_optimal_action_value.value;
});

// Map available actions to their values
var available_actions_to_values = cache(function(agentId, curr_iteration, gridworld, state_x, state_y) {
  // Map each possible action to its value
  return map(function(action) {
    // Calculate the utility of the current state and action
    var utility = utility_function(agentId, gridworld, state_x, state_y, action);
    // Get the next state based on the current action
    var next_state = gridworld_transition(gridworld, state_x, state_y, action);
    // Get the value of the next state
    var next_state_value = value_function(agentId, curr_iteration, gridworld, next_state.x, next_state.y);
    // Return the action and its total value (utility + value of next state)
    return { action: action, value: utility + next_state_value };
  }, directions);
});

// Determine the optimal action and its value
var optimal_action_value = cache(function(agentId, curr_iteration, gridworld, state_x, state_y) {
  // Get the values of all possible actions
  var actions_to_values = available_actions_to_values(agentId, curr_iteration, gridworld, state_x, state_y);
  // Return the action with the maximum value
  return maxBy(function(a) { return a.value; }, actions_to_values);
});






// Define the maximum number of iterations for planning
var MAX_ITERATIONS = 50;

// Determine if the agent should terminate
var should_terminate = cache(function(agentId, gridworld, state_x, state_y) {
  // If the value function is non-positive, terminate
  if (value_function(agentId, MAX_ITERATIONS, gridworld, initial_x, initial_y) <= 0) return true;
  // Get the location type of the current state
  var location_type = get_gridworld_at(gridworld, state_x, state_y);
  // Calculate the location utility
  var state_location_utility = location_utility(agentId, location_type);
  // Terminate if location utility is positive
  return state_location_utility > 0;
});

// Define the optimal policy from the initial state
var optimal_policy_from_initial_state = cache(function(agentId, gridworld, state_x, state_y) {
  // Check if the agent should terminate
  if (should_terminate(agentId, gridworld, state_x, state_y)) return [];
  // Get the optimal action and its value
  var curr_optimal_action_value = optimal_action_value(agentId, MAX_ITERATIONS, gridworld, state_x, state_y);
  var curr_optimal_action = curr_optimal_action_value.action;
  // Transition to the next state based on the optimal action
  var next_state = gridworld_transition(gridworld, state_x, state_y, curr_optimal_action);
  // Recursively find the remaining policy from the next state
  var remaining_policy = optimal_policy_from_initial_state(agentId, gridworld, next_state.x, next_state.y);
  // Return the optimal action concatenated with the remaining policy
  return [curr_optimal_action].concat(remaining_policy);
});

// Define the trajectory from the initial state
var trajectory_from_initial_state = cache(function(agentId, gridworld, state_x, state_y) {
  // Check if the agent should terminate
  if (should_terminate(agentId, gridworld, state_x, state_y)) return [];
  // Get the optimal action and its value
  var curr_optimal_action_value = optimal_action_value(agentId, MAX_ITERATIONS, gridworld, state_x, state_y);
  var curr_optimal_action = curr_optimal_action_value.action;
  // Transition to the next state based on the optimal action
  var next_state = gridworld_transition(gridworld, state_x, state_y, curr_optimal_action);
  // Recursively find the remaining trajectory from the next state
  var remaining_trajectory = trajectory_from_initial_state(agentId, gridworld, next_state.x, next_state.y);
  // Return the next state location concatenated with the remaining trajectory
  return [next_state.location].concat(remaining_trajectory);
});

// Define the optimal policy
var optimal_policy = cache(function(agentId, gridworld, initial_state_x, initial_state_y) {
  // Return the starting state and the optimal policy from the initial state
  return [['start', 'start']].concat(optimal_policy_from_initial_state(agentId, gridworld, initial_state_x, initial_state_y));
});

// Define the optimal trajectory
var optimal_trajectory = cache(function(agentId, gridworld, initial_state_x, initial_state_y) {
  // Return the starting location and the trajectory from the initial state
  return [get_gridworld_at(gridworld, initial_state_x, initial_state_y)].concat(trajectory_from_initial_state(agentId, gridworld, initial_state_x, initial_state_y));
});

// Define the optimal policy with trajectory
var optimal_policy_with_trajectory = cache(function(agentId, gridworld, initial_state_x, initial_state_y) {
  // Zip together the optimal policy and the optimal trajectory
  return zip(optimal_policy(agentId, gridworld, initial_state_x, initial_state_y), optimal_trajectory(agentId, gridworld, initial_state_x, initial_state_y));
});

// Get the terminal goal state
var get_terminal_goal_state = cache(function(agentId, gridworld, initial_state_x, initial_state_y) {
  return last(optimal_trajectory(agentId, gridworld, initial_state_x, initial_state_y));
});

// Check if the trajectory contains a specific location type
var trajectory_has_location_type = cache(function(agentId, locationType, gridworld, initial_state_x, initial_state_y) {
  return member(locationType, optimal_trajectory(agentId, gridworld, initial_state_x, initial_state_y));
});

// Check if the policy contains a specific motion type
var policy_has_motion_type = cache(function(agentId, motionType, gridworld, initial_state_x, initial_state_y) {
  var policy_motions = map(function(action) { return action[0]; }, optimal_policy(agentId, gridworld, initial_state_x, initial_state_y));
  return member(motionType, policy_motions);
});

// Check if the policy and trajectory contain a specific motion type at a location
var policy_and_trajectory_has_motion_at_location = cache(function(agentId, motionType, locationType, gridworld, initial_state_x, initial_state_y) {
  var policy_motions = map(function(action) { return action[0]; }, optimal_policy(agentId, gridworld, initial_state_x, initial_state_y));
  var trajectory = optimal_trajectory(agentId, gridworld, initial_state_x, initial_state_y);
  var motions_at_locations = zip(policy_motions, trajectory);
  return member([motionType, locationType], motions_at_locations);
});

// Get the motion at a location
var motion_at_location = cache(function(agentId, motionType, locationType, gridworld, initial_state_x, initial_state_y) {
  var policy_motions = map(function(action) { return action[0]; }, optimal_policy(agentId, gridworld, initial_state_x, initial_state_y));
  var trajectory = optimal_trajectory(agentId, gridworld, initial_state_x, initial_state_y);
  return zip(policy_motions, trajectory);
});

// Derived predicates

// Generate unique action IDs
var action_id_gensym = makeGensym("action-");

// Define actions for going to a location
var is_going_to_actions = cache(function(agentId) {
  // Get the policy with trajectory
  var action_states = optimal_policy_with_trajectory(agentId, gridworld, initial_x, initial_y);
  // Get the final location of the trajectory
  var final_location = last(last(action_states));
  // Define the action for going to the final location
  return [[
    ['action_id', action_id_gensym()],
    ['action_subject', agentId],
    ['action_predicates', ['is_going', ['to', final_location]]],
    ['action_preposition', 'to'],
    ['action_location', final_location]
  ]];
});

// Define actions for going on a route
var is_going_on_actions = cache(function(agentId) {
  // Get the policy with trajectory
  var action_states = optimal_policy_with_trajectory(agentId, gridworld, initial_x, initial_y);
  // Fold over the action states to define actions for each step
  return fold(function(action_state, these_actions) {
    var action_location = last(action_state);
    var action_manner = action_state[0][0];
    var action_direction = action_state[0][1];
    // Define the action for the current step
    return [
      [
        ['action_id', action_id_gensym()],
        ['action_subject', agentId],
        ['action_predicates', ['is_going', action_manner, action_direction, ['on', action_location]]],
        ['action_preposition', 'on'],
        ['action_location', action_location]
      ]
    ].concat(these_actions);
  }, [], action_states);
});

// Combine all actions in the scene
var actions_in_scene = cache(function(agentId) {
  // Combine actions for going to a location and going on a route
  return is_going_to_actions(agentId).concat(is_going_on_actions(agentId));
});

// Check if an action has a specific predicate
var is_action = function(action, action_predicate) {
  // Check if the action predicate is in the action's predicates
  return member(action_predicate, lookup(action, 'action_predicates'));
};

// Check if an entity is the subject of an action
var is_subject_of_action = function(action, entity) {
  // Check if the action's subject matches the entity
  return lookup(action, 'action_subject') === entity;
};

// Check if an action has a specific preposition
var is_preposition_of_action = function(action, preposition) {
  // Check if the action's preposition matches the given preposition
  return lookup(action, 'action_preposition') === preposition;
};

// Check if an action occurs at a specific location
var is_location_of_action = function(action, location) {
  // Check if the action's location matches the given location
  return lookup(action, 'action_location') === location;
};

// Get the location of an action
var get_location = function(action) {
  // Return the location of the action
  return lookup(action, 'action_location');
};

// Check if any action satisfies a predicate
var exists_action = function(agentId, predicate) {
  // Check if any action in the scene satisfies the predicate
  return some(map(predicate, actions_in_scene(agentId)));
};

// Get all actions that satisfy a predicate
var get_actions = function(agentId, predicate) {
  // Fold over actions in the scene to collect those that satisfy the predicate
  return fold(function(action, these_actions) {
    if (predicate(action)) return [action].concat(these_actions);
    return these_actions;
  }, [], actions_in_scene(agentId));
});