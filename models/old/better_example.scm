;; Define the gridworld environment (10x10 grid)
(define gridworld (list
  (list 'sidewalk 'grass 'grass 'grass 'grass 'grass 'grass 'grass 'grass 'hospital)
  (list 'sidewalk 'grass 'grass 'grass 'grass 'grass 'grass 'grass 'grass 'grass)
  (list 'sidewalk 'grass 'grass 'grass 'grass 'grass 'grass 'grass 'grass 'grass)
  (list 'sidewalk 'grass 'grass 'grass 'grass 'grass 'grass 'grass 'grass 'grass)
  (list 'sidewalk 'grass 'grass 'grass 'grass 'grass 'grass 'grass 'grass 'grass)
  (list 'sidewalk 'sidewalk 'sidewalk 'sidewalk 'sidewalk 'sidewalk 'sidewalk 'sidewalk 'sidewalk 'coffee_shop)
  (list 'sidewalk 'grass 'grass 'grass 'grass 'grass 'grass 'grass 'grass 'sidewalk)
  (list 'sidewalk 'grass 'grass 'grass 'grass 'grass 'grass 'grass 'grass 'sidewalk)
  (list 'sidewalk 'grass 'grass 'grass 'grass 'grass 'grass 'grass 'grass 'sidewalk)
  (list 'sidewalk 'sidewalk 'sidewalk 'sidewalk 'sidewalk 'sidewalk 'sidewalk 'sidewalk 'sidewalk 'sidewalk)))

;; Define the utilities for different goals. Hospital is high utility, coffee_shop is low
(define goal_utilities (list (cons 'hospital 100) (cons 'coffee_shop 25)))

;; Function to get the type of tile at a given position
(define (get-tile-type grid pos)
  (list-ref (list-ref grid (second pos)) (first pos)))

;; Function to calculate the utility of reaching a goal
(define (goal-utility pos)
  (let ((tile-type (get-tile-type gridworld pos)))
    (cdr (assoc tile-type goal_utilities))))

;; Function to return the minimum steps to the goal based on hard-coded optimal paths
(define (min-steps-to-goal goal)
  (cond ((eq? goal 'coffee_shop) 14) ;; Shortest path to coffee shop
        ((eq? goal 'hospital) 9)))  ;; Shortest path to hospital

;; Function to calculate the enforcer's penalty based on the agent's path
(define (calculate-enforcer-penalty path goal)
  (let* ((optimal-steps (min-steps-to-goal goal))
         (grass-penalty (reduce + 0 (map (lambda (step)
                                           (if (equal? (get-tile-type gridworld step) 'grass) -1 0))
                                         path))))
    grass-penalty))

;; Function to calculate the penalty for a given trajectory
(define (calculate-penalty walker-utility enforcer-penalty path-length optimal-steps)
  (if (= enforcer-penalty 0)
      0  ;; No penalty if the enforcer didn't step on grass
      (let* ((grass-proportion (/ (abs enforcer-penalty) path-length))  ;; Calculate the proportion of the path on grass
             (path-ratio (/ path-length optimal-steps))  ;; Calculate the ratio of actual path length to optimal path length
             (suboptimality-factor (- path-ratio 1))  ;; Calculate the suboptimality factor
             (suboptimality-weight 2)  ;; Weight for the suboptimality factor
             (penalty (* 100 (+ grass-proportion (* suboptimality-weight suboptimality-factor)))))  ;; Calculate the penalty based on grass proportion and weighted suboptimality factor
        (exact->inexact (min 100 penalty)))))  ;; Cap the penalty at 100 and convert to a decimal number

;; Function to calculate the fine for a given trajectory
(define (calculate-fine trajectory)
  (let* ((final-pos (last trajectory))
         (goal (get-tile-type gridworld final-pos))  ;; Get the tile type of the final position to determine the goal
         (walker-utility (goal-utility final-pos))
         (enforcer-penalty (calculate-enforcer-penalty trajectory goal))
         (path-length (length trajectory))
         (optimal-steps (min-steps-to-goal goal)))  ;; Get the optimal number of steps to the goal
    (calculate-penalty walker-utility enforcer-penalty path-length optimal-steps)))



;; Function to analyze trajectory and print tile counts and percentages
(define (analyze-trajectory-stats trajectory)
  (let* ((trajectory-minus-end (butlast trajectory))  ; Remove the last step (destination)
         (grass-count (count-tile-type gridworld trajectory-minus-end 'grass))
         (sidewalk-count (count-tile-type gridworld trajectory-minus-end 'sidewalk))
         (total-non-end-tiles (length trajectory-minus-end))
         (grass-percentage (if (> total-non-end-tiles 0)
                               (* (/ (exact->inexact grass-count) (exact->inexact total-non-end-tiles)) 100)  ; Convert to floating point and calculate percentage
                               0))) ; Calculate percentage of grass
    ;; Print the statistics
    (display "Grass count: ") (display grass-count)
    (display ", Sidewalk count: ") (display sidewalk-count)
    (display ", Percentage of grass: ") (display (round grass-percentage)) (display "%")  ; Round to nearest whole number
    (newline)))

;; Helper function to count occurrences of a tile type in a trajectory
(define (count-tile-type grid trajectory tile-type)
  (let ((count (fold-left (lambda (sum step)
                            (if (equal? (get-tile-type grid step) tile-type)
                                (+ sum 1)
                                sum))
                          0
                          trajectory)))
    count))

;; Function to remove the last element from a list because we don't want to count the destination in the statistics
(define (butlast lst)
  (if (null? (cdr lst))
      '()
      (cons (car lst) (butlast (cdr lst)))))


;; Example trajectory to the coffee shop
(define example-trajectory-coffee-optimal-sidewalk '((0 0) (0 1) (0 2) (0 3) (0 4) (0 5) (1 5) (2 5) (3 5) (4 5) (5 5) (6 5) (7 5) (8 5) (9 5)))
(define example-trajectory-coffee-optimal-grass '((0 0) (0 1) (0 2) (0 3) (0 4) (1 4) (2 4) (3 4) (4 4) (5 4) (6 4) (7 4) (8 4) (9 4) (9 5)))
(define example-trajectory-coffee-suboptimal-grass '((0 0) (0 1) (0 2) (0 3) (0 4) (0 5) (0 6) (1 6) (2 6) (3 6) (4 6) (5 6) (6 6) (7 6) (8 6) (9 6) (9 5)))
(define example-trajectory-coffee-suboptimal-sidewalk '((0 0) (0 1) (0 2) (0 3) (0 4) (0 5) (0 6) (0 7) (0 8) (0 9) (1 9) (2 9) (3 9) (4 9) (5 9) (6 9) (7 9) (8 9) (9 9) (9 8) (9 7) (9 6) (9 5)))

;; Calculate the fine for this trajectory
(define fine-coffee-optimal-sidewalk (calculate-fine example-trajectory-coffee-optimal-sidewalk))
(define fine-coffee-optimal-grass (calculate-fine example-trajectory-coffee-optimal-grass))
(define fine-coffee-suboptimal-grass (calculate-fine example-trajectory-coffee-suboptimal-grass))
(define fine-coffee-suboptimal-sidewalk (calculate-fine example-trajectory-coffee-suboptimal-sidewalk))

;; Print the results
(newline)
(display "Fine for the trajectory to the coffee shop (optimal sidewalk): ")
(display fine-coffee-optimal-sidewalk)
(newline)
(analyze-trajectory-stats example-trajectory-coffee-optimal-sidewalk)
(newline)

(display "Fine for the trajectory to the coffee shop (optimal grass): ")
(display fine-coffee-optimal-grass)
(newline)
(analyze-trajectory-stats example-trajectory-coffee-optimal-grass)
(newline)

(display "Fine for the trajectory to the coffee shop (suboptimal grass): ")
(display fine-coffee-suboptimal-grass)
(newline)
(analyze-trajectory-stats example-trajectory-coffee-suboptimal-grass)
(newline)

(display "Fine for the trajectory to the coffee shop (suboptimal sidewalk): ")
(display fine-coffee-suboptimal-sidewalk)
(newline)
(analyze-trajectory-stats example-trajectory-coffee-suboptimal-sidewalk)
(newline)