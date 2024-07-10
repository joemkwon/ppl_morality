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
  (list 'sidewalk 'sidewalk 'sidewalk 'sidewalk 'sidewalk 'sidewalk 'sidewalk 'sidewalk 'sidewalk 'coffee_shop)))

;; Define the utilities for different goals. Hospital is high utility, coffee_shop is low
(define goal_utilities (list (cons 'hospital 100) (cons 'coffee_shop 25)))

;; Function to get the type of tile at a given position
(define (get-tile-type grid pos)
  (list-ref (list-ref grid (second pos)) (first pos)))

;; Function to calculate the utility of reaching a goal
(define (goal-utility pos)
  (let ((tile-type (get-tile-type gridworld pos)))
    (cdr (assoc tile-type goal_utilities))))

;; Function to calculate the enforcer's penalty based on the agent's path
(define (calculate-enforcer-penalty path)
  (reduce + 0 (map (lambda (step)
                     (if (equal? (get-tile-type gridworld step) 'grass) -1 0))
                   path)))

;; Function to calculate the penalty for a given trajectory
;; For example, if the walker goes to hospital (100 utility) but steps on grass 5 times,
;; the penalty would be 5% of 100 = 5. 
;;(walker-utility 100, enforcer-penalty -5 but absoulte value is 5), then ratio is 5/100 or 0.05 and penalty is 0.05*100 = 5 dollars
(define (calculate-penalty walker-utility enforcer-penalty)
  (if (= enforcer-penalty 0)
      0 ; No penalty if the enforcer didn't step on grass
      (let* ((ratio (/ (abs enforcer-penalty) walker-utility)) ; Calculate the ratio of enforcer penalty to walker utility
             (penalty (* 100 ratio))) ; Calculate a raw penalty as a direct percentage of utility
        (min 100 penalty)))) ; Cap the penalty at 100 dollars

;; Function to calculate the penalty for a given trajectory
;; Use final position to get the utility of goal reached, then sum the enforcer penalties for rule violations along path
(define (calculate-fine trajectory)
  (let ((final-pos (last trajectory))
        (walker-utility (goal-utility (last trajectory))) ; Utility of the goal reached by walker
        (enforcer-penalty (calculate-enforcer-penalty trajectory))) ; Penalty for enforcer stepping on grass
    (calculate-penalty walker-utility enforcer-penalty)))



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


;; Example trajectory 
(define example-trajectory-hospital-somegrass '((0 0) (0 1) (1 1) (2 1) (3 1) (4 1) (5 1) (6 1) (7 1) (8 1) (9 1) (9 0)))
(define example-trajectory-hospital-nograss '((0 0) (1 0) (2 0) (3 0) (4 0) (5 0) (6 0) (7 0) (8 0) (9 0)))
(define example-trajectory-hospital-lotsgrass '((0 0) (0 1) (1 1) (2 1) (2 2) (2 3) (2 4) (2 3) (2 2) (2 1) (2 2) (3 2) (3 1) (4 1) (5 1) (6 1) (7 1) (8 1) (9 1) (9 0)))

;; Calculate the fine for the example trajectories of going to the hospital
(define fine0 (calculate-fine example-trajectory-hospital-nograss))
(define fine1 (calculate-fine example-trajectory-hospital-somegrass))
(define fine2 (calculate-fine example-trajectory-hospital-lotsgrass))


;; Print the results
(newline)
(display "Fine for trajectory to hospital with no grass stepped on: ")
(display fine0)
(newline)
(analyze-trajectory-stats example-trajectory-hospital-nograss)
(newline)

(display "Fine for trajectory to hospital with some grass stepped on: ")
(display fine1)
(newline)
(analyze-trajectory-stats example-trajectory-hospital-somegrass)
(newline) 

(display "Fine for trajectory to hospital with lots of grass stepped on: ")
(display fine2)
(newline)
(analyze-trajectory-stats example-trajectory-hospital-lotsgrass)
(newline)

;; Calculate the fine for the example trajectories of going to the coffee shop
(define example-trajectory-coffee-nograss '((0 0) (0 1) (0 2) (0 3) (0 4) (0 5) (0 6) (0 7) (0 8) (0 9) (1 9) (2 9) (3 9) (4 9) (5 9) (6 9) (7 9) (8 9) (9 9)))
(define example-trajectory-coffee-somegrass '((0 0) (0 1) (0 2) (0 3) (0 4) (0 5) (0 6) (0 7) (1 7) (1 8) (1 9) (2 9) (3 9) (4 9) (5 9) (6 9) (7 9) (8 9) (9 9)))
(define example-trajectory-coffee-lotsgrass '((0 0) (0 1) (1 1) (1 2) (2 2) (2 3) (3 3) (3 4) (4 4) (4 5) (5 5) (5 6) (6 6) (6 7) (7 7) (7 8) (8 8) (8 9) (9 9)))

;; Calculate the fine for the example trajectories of going to the coffee shop
(define fine3 (calculate-fine example-trajectory-coffee-nograss))
(define fine4 (calculate-fine example-trajectory-coffee-somegrass))
(define fine5 (calculate-fine example-trajectory-coffee-lotsgrass))

;; Print the results
(newline)
(display "Fine for trajectory to coffee shop with no grass stepped on: ")
(display fine3)
(newline)
(analyze-trajectory-stats example-trajectory-coffee-nograss)
(newline)

(display "Fine for trajectory to coffee shop with some grass stepped on: ")
(display fine4)
(newline)
(analyze-trajectory-stats example-trajectory-coffee-somegrass)
(newline)

(display "Fine for trajectory to coffee shop with lots of grass stepped on: ")
(display fine5)
(newline)
(analyze-trajectory-stats example-trajectory-coffee-lotsgrass)
(newline)