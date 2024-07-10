;; Define the gridworld environment (10x10 grid)
(define gridworld (list
  (list 'sidewalk 'sidewalk 'sidewalk 'sidewalk 'sidewalk 'sidewalk 'sidewalk 'sidewalk 'sidewalk 'hospital)
  (list 'sidewalk 'grass 'grass 'grass 'grass 'grass 'grass 'grass 'grass 'sidewalk)
  (list 'sidewalk 'grass 'grass 'grass 'grass 'grass 'grass 'grass 'grass 'sidewalk)
  (list 'sidewalk 'grass 'grass 'grass 'grass 'grass 'grass 'grass 'grass 'sidewalk)
  (list 'sidewalk 'grass 'grass 'grass 'grass 'grass 'grass 'grass 'grass 'sidewalk)
  (list 'sidewalk 'grass 'grass 'grass 'grass 'grass 'grass 'grass 'grass 'sidewalk)
  (list 'sidewalk 'grass 'grass 'grass 'grass 'grass 'grass 'grass 'grass 'sidewalk)
  (list 'sidewalk 'grass 'grass 'grass 'grass 'grass 'grass 'grass 'grass 'sidewalk)
  (list 'sidewalk 'grass 'grass 'grass 'grass 'grass 'grass 'grass 'grass 'sidewalk)
  (list 'sidewalk 'sidewalk 'sidewalk 'sidewalk 'sidewalk 'sidewalk 'sidewalk 'sidewalk 'sidewalk 'coffee_shop)
))

;; Starting position of the agent
(define start '(0 0))

;; Size of the grid
(define grid-size 10)

;; Define the agent's possible actions
(define actions '(up down left right))

;; Define the goals
(define goals (list 'coffee_shop 'hospital))

;; Define the utilities for different goals. Hospital is high utility, coffee_shop is low
(define goal_utilities (list (cons 'hospital 100) (cons 'coffee_shop 25)))

;; Function to get the type of tile at a given position
(define (get-tile-type grid pos)
  (let ((x (first pos))
        (y (second pos)))
    (list-ref (list-ref grid y) x)))

;; Function to move the agent to a new position based on the action
(define (move pos action)
  (match action
    ('up (list (first pos) (max 0 (- (second pos) 1))))
    ('down (list (first pos) (min (- grid-size 1) (+ (second pos) 1))))
    ('left (list (max 0 (- (first pos) 1)) (second pos)))
    ('right (list (min (- grid-size 1) (+ (first pos) 1)) (second pos)))))

;; Function to calculate the utility of reaching a goal
(define (goal-utility pos)
  (let ((tile-type (get-tile-type gridworld pos)))
    (cdr (assoc tile-type goal_utilities))))

;; Define the agent's decision-making process to move directly towards the goal
(define (choose-action pos goal)
  (let ((goal-x (first goal))
        (goal-y (second goal))
        (pos-x (first pos))
        (pos-y (second pos)))
    (if (< pos-x goal-x) 'right
        (if (> pos-x goal-x) 'left
            (if (< pos-y goal-y) 'down
                (if (> pos-y goal-y) 'up
                    'stay))))))

;; Function to simulate the agent's path from start to a goal
(define (simulate-path pos path goal)
  (if (equal? (get-tile-type gridworld pos) goal)
      (list path (goal-utility pos))
      (let ((action (choose-action pos goal)))
        (simulate-path (move pos action) (cons (list pos action) path) goal))))

;; Function to calculate the enforcer's penalty based on the agent's path
(define (calculate-enforcer-penalty path)
  (reduce + (map (lambda (step)
    (if (equal? (get-tile-type gridworld (car step)) 'grass) -1 0)) path)))

;; Function to calculate the penalty in dollars based on the ratio of walker utility to enforcer penalty
(define (calculate-penalty walker-utility enforcer-penalty)
  (if (= enforcer-penalty 0)
      0
      (let ((ratio (/ walker-utility (abs enforcer-penalty))))
        (min 100 (* 100 (exp (- ratio)))))))

;; Function to run the simulation for a given goal and calculate the penalty
(define (simulate-and-calculate-penalty goal)
  (let ((result (simulate-path start '() goal)))
    (let ((walker-utility (cadr result))
          (enforcer-penalty (calculate-enforcer-penalty (car result))))
      (calculate-penalty walker-utility enforcer-penalty))))

;; Run the simulation for all goals and calculate penalties
(define penalties (map (lambda (goal)
  (cons goal (simulate-and-calculate-penalty goal))) goals))

;; Print the results
penalties
