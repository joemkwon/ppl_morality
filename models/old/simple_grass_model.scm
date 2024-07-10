;; Define the gridworld environment
(define gridworld (list
  (list 'sidewalk 'grass 'sidewalk 'hospital)
  (list 'sidewalk 'sidewalk 'grass 'sidewalk)
  (list 'sidewalk 'grass 'sidewalk 'coffee_shop)
  (list 'sidewalk 'sidewalk 'sidewalk 'sidewalk)))

(define start '(0 0))

;; Define the agent's possible actions
(define actions '(up down left right))

;; Define the utilities for different goals
(define goal_utilities (list (cons 'hospital 10) (cons 'coffee_shop 5)))

;; Function to get the type of tile at a given position
(define (get-tile-type grid pos)
  (let ((x (first pos))
        (y (second pos)))
    (list-ref (list-ref grid y) x)))

;; Function to move the agent to a new position based on the action
(define (move pos action)
  (match action
    ('up (list (first pos) (max 0 (- (second pos) 1))))
    ('down (list (first pos) (min 3 (+ (second pos) 1))))
    ('left (list (max 0 (- (first pos) 1)) (second pos)))
    ('right (list (min 3 (+ (first pos) 1)) (second pos)))))

;; Function to calculate the utility of reaching a goal
(define (goal-utility pos)
  (let ((tile-type (get-tile-type gridworld pos)))
    (cdr (assoc tile-type goal_utilities))))

;; Function to calculate the utility of a move based on the tile type
(define (move-utility pos action)
  (let ((new-pos (move pos action))
        (tile-type (get-tile-type gridworld new-pos)))
    (if (equal? tile-type 'grass) -1 0)))

;; Define the agent's decision-making process using a probabilistic model
(define (choose-action pos)
  (categorical (map (lambda (action) (exp (move-utility pos action))) actions)))

;; Function to simulate the agent's path from start to a goal
(define (simulate-path pos path goal)
  (if (equal? (get-tile-type gridworld pos) goal)
      (list path (goal-utility pos))
      (let ((action (choose-action pos)))
        (simulate-path (move pos action) (cons (list pos action) path) goal))))

;; Enforcer penalizes grass walking
(define (enforcer-penalty path)
  (reduce + (map (lambda (step)
    (if (equal? (get-tile-type gridworld (move (car step) (cdr step))) 'grass) -1 0)) path)))

;; Run the simulation for both goals
(define result-hospital (simulate-path start '() 'hospital))
(define result-coffee-shop (simulate-path start '() 'coffee_shop))

;; Calculate total utilities considering the enforcer's penalty
(define total-utility-hospital (+ (cadr result-hospital) (enforcer-penalty (car result-hospital))))
(define total-utility-coffee-shop (+ (cadr result-coffee-shop) (enforcer-penalty (car result-coffee-shop))))

;; Print the results
(list (cons 'hospital total-utility-hospital) (cons 'coffee_shop total-utility-coffee-shop))
