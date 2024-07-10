var model = function() {
    var goalProbabilities = { 'H': 0.1, 'C': 0.9 };  // Prior probabilities for goals
    var rule_exists = true; // does the rule exist or not

    // utilities for the 3 types of terrains and 2 goals
    var terrainUtilities = {
        'S': {utility: gaussian(-1, 0.5)},
        'G': {utility: rule_exists ? gaussian(-5, 1) : gaussian(-1, 0.5)},
        'T': {utility: gaussian(-10, 1)},

        'H': {utility: gaussian(70, 15)},
        'C': {utility: gaussian(30, 10)}
    };


    // the grid
    var grid = [
        ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'T', 'T', 'H'],
        ['S', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G'],
        ['S', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G'],
        ['S', 'G', 'G', 'G', 'G', 'T', 'G', 'G', 'G', 'G'],
        ['S', 'G', 'G', 'G', 'T', 'T', 'G', 'G', 'G', 'G'],
        ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'C'],
        ['S', 'G', 'T', 'T', 'G', 'G', 'G', 'G', 'G', 'S'],
        ['S', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'S'],
        ['S', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'S'],
        ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S']
    ];

    // Observed partial trajectory of movement
    var observed_trajectory = [{x: 0, y: 0}, {x: 0, y: 1}, {x: 0, y: 2}, {x: 0, y: 3}, {x: 0, y: 4}, {x: 0, y: 5}];

    // action and state transitions
    var actions = [{dx: 0, dy: 1}, {dx: 1, dy: 0}, {dx: 0, dy: -1}, {dx: -1, dy: 0}];

    // the transition model
    var transition = function(state, action) {
        var x = state.x + action.dx;
        var y = state.y + action.dy;
        if (x < 0 || x >= grid.length || y < 0 || y >= grid[0].length) {
            return state;
        }
        return {x: x, y: y};
    };

    // the utility function

    // Simulate an agent from the starting coordinate to a sampled goal 1000 times?

    // See how many of the sampled trajectories match the partial observed trajectory

    // Update agent's utility function based on the proportion of sampled trajectories containing the observed trajectory?

    // Return the updated utility function?
}