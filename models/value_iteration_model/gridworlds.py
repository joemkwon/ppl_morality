# We'll use this file to just store all the various maps we come up with for the stimuli

def gridworld_1():
    return [
        ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S'],
        ['S', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'S'],
        ['S', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'S'],
        ['S', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'S'],
        ['S', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'S'],
        ['S', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'S'],
        ['S', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'S'],
        ['S', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'S'],
        ['S', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'S'],
        ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S']
    ]


GRIDWORLDS = {
    'gridworld_1': gridworld_1,
}
