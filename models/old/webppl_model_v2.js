// Define the grid
var grid = [
  ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'H'],
  ['S', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G'],
  ['S', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G'],
  ['S', 'G', 'G', 'G', 'G', 'S', 'G', 'G', 'G', 'G'],
  ['S', 'G', 'G', 'G', 'S', 'S', 'G', 'G', 'G', 'G'],
  ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'C'],
  ['S', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'S'],
  ['S', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'S'],
  ['S', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'S'],
  ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S']
];

// Stochastic properties and parameters
var initialState = {
  maxGrassTraffic: 0,
  grassDead: false,
  utility: 0
};

// Functions for utility and cost
function getTerrainUtility(x, y) {
  var terrain = grid[y][x];
  switch(terrain) {
    case 'H': return 1000; // Hospital utility
    case 'C': return 100;  // Coffee shop utility
    default: return 0;
  }
}

function getTerrainCost(x, y) {
  return grid[y][x] === 'S' || grid[y][x] === 'G' ? -1 : 0;
}

// Search for all goal coordinates
var hospitalCoords = [];
var coffeeShopCoords = [];
for (var y = 0; y < grid.length; y++) {
  for (var x = 0; x < grid[y].length; x++) {
    if (grid[y][x] === 'H') hospitalCoords.push({x, y});
    else if (grid[y][x] === 'C') coffeeShopCoords.push({x, y});
  }
}

// Check if coordinates match any goal
function isGoalReached(goal, x, y) {
  if (goal === 'H') {
    return hospitalCoords.some(coord => coord.x === x && coord.y === y);
  } 
  else if (goal === 'C') {
    return coffeeShopCoords.some(coord => coord.x === x && coord.y === y);
  }
  return false;
}

// Simulation of agents
var numPeople = Math.round(gaussian(100, 10));
var peopleGoals = repeat(numPeople, function() { return flip(0.05) ? 'H' : 'C'; });

// Set up agent structure
var makeMDPAgent = function(params, world) {
  var stateToActions = world.stateToActions;
  var transition = world.transition;
  var utility = params.utility;
  var alpha = params.alpha;

  var act = dp.cache(
    function(state) {
      return Infer({ model() {
        var action = uniformDraw(stateToActions(state));
        var eu = expectedUtility(state, action);
        factor(alpha * eu);
        return action;
      }});
    });

  var expectedUtility = dp.cache(
    function(state, action){
      var u = utility(state, action);
      return u + expectation(Infer({ model() {
        var nextState = transition(state, action);
        var nextAction = sample(act(nextState));
        return expectedUtility(nextState, nextAction);
      }}));
    });

  return { params, expectedUtility, act };
};

// Set up world
var makeGridWorld = function(options) {
  var localGrid = options.grid;
  var start = options.start;

  var stateToActions = function(state) {
    var actions = [
      { dx: 0, dy: -1 }, // Up
      { dx: 0, dy: 1 },  // Down
      { dx: -1, dy: 0 }, // Left
      { dx: 1, dy: 0 }   // Right
    ];

    var validActions = actions.filter(function(action) {
      var newX = state.x + action.dx;
      var newY = state.y + action.dy;
      return newX >= 0 && newX < localGrid[0].length && newY >= 0 && newY < localGrid.length;
    });

    return validActions;
  };

  var transition = function(state, action) {
    var newX = state.x + action.dx;
    var newY = state.y + action.dy;
    return {
      x: newX,
      y: newY
    };
  };

  var startState = {
    x: start[0],
    y: start[1]
  };

  return { grid: localGrid, stateToActions, transition, startState };
};

var utility = function(state, action, envState) {
  var terrain = grid[state.y][state.x];
  var baseUtilityValue = getTerrainUtility(state.x, state.y);
  var costValue = getTerrainCost(state.x, state.y);
  var penalty = 0;  // Initialize penalty to zero

  if (terrain === 'G') {
    var newGrassTraffic = envState.grassTraffic + 1;
    var newGrassDead = envState.grassDead;

    // Stochastic computation inside expected utility
    var maxGrassTraffic = gaussian(0.5, 0.1) * 50; // Sample grassFragility and compute

    if (!newGrassDead && newGrassTraffic > maxGrassTraffic) {
      var newGrassDead = true;
      var penalty = 1000; // Penalty for grass getting destroyed
    }

    // Calculate utility considering the potential penalty
    var totalUtility = baseUtilityValue + costValue - penalty;

    // Return the new state with adjusted values
    return {
      grassTraffic: newGrassTraffic,
      grassDead: newGrassDead,
      utility: totalUtility
    };
  } else {
    // No penalty, so compute utility normally
    var totalUtility = baseUtilityValue + costValue;

    // Return the new state with unchanged grass conditions
    return {
      grassTraffic: envState.grassTraffic,
      grassDead: envState.grassDead,
      utility: totalUtility
    };
  }
};

var agent = makeMDPAgent({ utility, alpha: 1000 }, makeGridWorld({
  grid: grid,
  start: [0, 0]
}));


var simulate = function(startState, world, agent, goal, maxSteps) {
  var act = agent.act;
  var transition = world.transition;
  var currentStep = 0;

  var sampleSequence = function(state) {
    if (isGoalReached(goal, state.x, state.y) || currentStep++ > maxSteps) {
      return [state]; // Termination condition
    }
    var action = sample(act(state));
    var nextState = transition(state, action);
    return [state].concat(sampleSequence(nextState));
  };

  return sampleSequence(startState);
};


var world = makeGridWorld({
  grid: grid,
  start: [0, 0]
});

var simulatePersonShortestPath = function(goal) {
  return simulate(world.startState, world, agent, goal);
};

var expectedUtilitiesShortestPath = map(simulatePersonShortestPath, peopleGoals);
var averageUtilityShortestPath = sum(expectedUtilitiesShortestPath) / numPeople;

console.log("Average Utility (Shortest Path): ", averageUtilityShortestPath);
