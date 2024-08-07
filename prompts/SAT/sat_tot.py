vote_prompt = '''Given a question and several choices of next steps, analyze each choice in detail and compare them to decide which choice is the most promising to be the next step to solve the question. After analyzing each choice in detail and comparing them, conclude your final choice with \"Therefore, the best choice is\".
Example:

Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Choice 1: There are 15 trees originally.
Choice 2: There are originally 3 cars.
Choice 3: There are 20 trees originally.
Response: Choice 1 logically follows the progression of the problem: recognizing the original amount. However, Choice 2 involves irrelevant information (3 cars) and Choice 3 contains incorrect information (20 trees). Therefore, the best choice is 1. 

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Choice 1: Start with 3 cars. 2 more cars arrive, making it 3 + 2 = 5.
Choice 2: There are 3 cars. If 3 more cars leave, it's 3 - 3 = 0.
Choice 3: Originally 3 cars. Double the number, 3 * 2 = 6.
Response: Choice 1 is the only one that accurately reflects the scenario: starting with the initial number of cars and adding the ones that arrive. Choices 2 and 3 involve operations that are not relevant to the situation described in the question. Therefore, the best choice is 1.

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Choice 1: Leah had 32 chocolates. Her sister had 42. Total is 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.
Choice 2: Leah had 32 chocolates. If she gives half to her sister, it's 32 / 2 = 16. Not relevant to the question.
Choice 3: They start with 32 and 42 chocolates. If they lose some, say 10, it's 32 - 10 = 22 and 42 - 10 = 32. Total 22 + 32 = 54.
Response: Choice 1 correctly calculates the total number of chocolates and then subtracts the amount they ate, providing the correct remaining total. Choices 2 and 3 involve calculations or scenarios that do not align with the question. Therefore, the best choice is 1.

Question: {instruction}
'''


cot_prompt = '''Imagine you are trying to solve a math problem with a step-by-step approach. At each step, you should propose a single next best step to solve the problem which may involve arithmetic calculation. Please start your answer with "Let's think step by step" in the very first sentence. When the original question is answerable, please start the subquestion with "The answer is".
IMPORTANT: MAKE SURE NOT TO HAVE THE DIRECT ANSWER IN YOUR POSSIBLE STEPS OUTPUT, JUST MAKE ONE STEP AT A TIME.
Solved Example:

Question: Simplify and write the result with a rational denominator: $$\sqrt{\sqrt[3]{\sqrt{\frac{1}{729}}}}$$
A. $\frac{3\sqrt{3}}{3}$
B. $\frac{1}{3}$ 
C. $\sqrt{3}$ 
D. $\frac{\sqrt{3}}{3}$
Step 1: Simplify the innermost radical first, $\sqrt{\frac{1}{729}}$: $$\sqrt{\frac{1}{729}} = \sqrt{\frac{1}{27^2}} = \frac{1}{\sqrt{27^2}} = \frac{1}{27}$$
Step 2: Next, simplify the cube root of $\frac{1}{27}$: $$\sqrt[3]{\frac{1}{27}} = \frac{1}{\sqrt[3]{27}} = \frac{1}{3}$$
Step 3: Finally, simplify the square root of $\frac{1}{3}$: $$\sqrt{\frac{1}{3}} = \frac{\sqrt{1}}{\sqrt{3}} = \frac{1}{\sqrt{3}}$$
Step 4: Rationalize the denominator: $$\frac{1}{\sqrt{3}} = \frac{1 \cdot \sqrt{3}}{\sqrt{3} \cdot \sqrt{3}} = \frac{\sqrt{3}}{3}$$
Step 5: The answer is $\frac{\sqrt{3}}{3}$.
Step 6: The answer is D.

Question: Five thousand dollars compounded annually at an $x\%$ interest rate takes six years to double. At the same interest rate, how many years will it take \$300 to grow to \$9600?
A. 12 
B. 1 
C. 30 
D. 5
Step 1: First, we need to determine the interest rate $x$. We know that \$5000 doubles to \$10000 in six years. Using the formula for compound interest: $$ A = P(1 + \frac{r}{100})^t $$ where $A$ is the final amount, $P$ is the principal amount, $r$ is the interest rate, and $t$ is the time in years. For the amount to double: $$ 10000 = 5000(1 + \frac{x}{100})^6 $$ $$ 2 = (1 + \frac{x}{100})^6 $$
Step 2: To find $x$, take the sixth root of both sides: $$ 1 + \frac{x}{100} = 2^{1/6} $$ $$ \frac{x}{100} = 2^{1/6} - 1 $$ $$ x = 100(2^{1/6} - 1) $$
Step 3: Now, using the same interest rate $x$, we need to find out how many years it will take for \$300 to grow to \$9600. $$ 9600 = 300(1 + \frac{x}{100})^t $$ $$ 32 = (1 + \frac{x}{100})^t $$
Step 4: Using the value $1 + \frac{x}{100} = 2^{1/6}$: $$ 32 = (2^{1/6})^t $$ $$ 32 = 2^{t/6} $$ Since $32 = 2^5$: $$ 2^5 = 2^{t/6} $$ $$ 5 = \frac{t}{6} $$ $$ t = 30 $$
Step 5: The answer is C.

Question: Ten students take a biology test and receive the following scores: 45, 55, 50, 70, 65, 80, 40, 90, 70, 85. What is the mean of the students test scores?
A. 55 
B. 60 
C. 62 
D. 65
Step 1: Add all the scores together: $45 + 55 + 50 + 70 + 65 + 80 + 40 + 90 + 70 + 85 = 650$
Step 2: Divide the total by the number of students: $\frac{650}{10} = 65$
Step 3: The answer is D.

Question: The variable $x$ varies directly as the square of $y$, and $y$ varies directly as the cube of $z$. If $x$ equals $-16$ when $z$ equals 2, what is the value of $x$ when $z$ equals \(\frac{1}{2}\)?
A. -1 
B. 16 
C. -\frac{1}{256} 
D. \frac{1}{16}
Step 1: Since $x$ varies directly as the square of $y$, we can write $x = k_1 y^2$ for some constant $k_1$.
Step 2: Since $y$ varies directly as the cube of $z$, we can write $y = k_2 z^3$ for some constant $k_2$.
Step 3: Substituting $y = k_2 z^3$ into $x = k_1 y^2$, we get $x = k_1 (k_2 z^3)^2 = k_1 k_2^2 z^6$.
Step 4: We know that $x = -16$ when $z = 2$. Substituting these values in, we get $-16 = k_1 k_2^2 (2^6)$. 
Step 5: Simplifying, $-16 = k_1 k_2^2 \cdot 64$.
Step 6: Solving for $k_1 k_2^2$, we get $k_1 k_2^2 = \frac{-16}{64} = -\frac{1}{4}$.
Step 7: Now we need to find the value of $x$ when $z = \frac{1}{2}$. Using $x = k_1 k_2^2 z^6$ and substituting $k_1 k_2^2 = -\frac{1}{4}$ and $z = \frac{1}{2}$, we get $x = -\frac{1}{4} (\frac{1}{2})^6$.
Step 8: Simplifying, $x = -\frac{1}{4} \cdot \frac{1}{64} = -\frac{1}{256}$.
Step 9: The answer is C.

Question: Joe was in charge of lights for a dance. The red light blinks every two seconds, the yellow light every three seconds, and the blue light every five seconds. If we include the very beginning and very end of the dance, how many times during a seven minute dance will all the lights come on at the same time? (Assume that all three lights blink simultaneously at the very beginning of the dance.)
A. 3 
B. 15 
C. 6 
D. 5
Step 1: Find the least common multiple (LCM) of the blinking intervals of the red, yellow, and blue lights. The LCM of 2, 3, and 5 is 30 seconds.
Step 2: Determine the total duration of the dance in seconds: 7 minutes = 7 \times 60 = 420 seconds.
Step 3: Calculate how many times the lights blink together in 420 seconds. They blink together every 30 seconds.
Step 4: Divide the total duration by the LCM: \(\frac{420}{30} = 14\).
Step 5: Include the very beginning of the dance, so the total number of times is 14 + 1 = 15.
Step 6: The answer is B.

Question: {input}
'''

vote_prompt_2 = '''Given a question and several choices of next steps, decide which choice is the most promising to solve the question. Analyze each choice in detail, then conclude in the last line "The best choice is s", where s the integer id of the choice.
Instruction: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Choice 1: There are 15 trees originally.
Choice 2: There are originally 3 cars.333
Choice 3: Originally, Leah had 32 chocolates.
Response: The best choice is 1.

Instruction: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Choice 1: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74.
Choice 2: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. 3 + 2 = 5.
Response: The best choice is 2.

Instruction: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Choice 1: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.
Response: The best choice is 1.

Insurction: {instruction}
'''

cot_prompt_2 = '''Imagine you are trying to solve a math problem with a step-by-step approach. At each step, you should propose a single next best step to solve the problem which may involve arithmetic calculation. When the original question is answerable, please start the subquestion with \"The answer is\".
IMPORTANT: MAKE SURE NOT TO HAVE THE DIRECT ANSWER IN YOUR POSSIBLE STEPS OUTPUT, JUST MAKE ONE STEP AT A TIME.
Solved Example:

Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Steps:
There are 15 trees originally.
Then there were 21 trees after some more were planted.
So there must have been 21 - 15 = 6.
The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Steps:
There are originally 3 cars.
2 more cars arrive.
3 + 2 = 5. 
The answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Steps:
Originally, Leah had 32 chocolates.
Her sister had 42.
So in total they had 32 + 42 = 74.
After eating 35, they had 74 - 35 = 39.
The answer is 39.

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Steps:
Jason started with 20 lollipops.
Then he had 12 after giving some to Denny.
So he gave Denny 20 - 12 = 8.
The answer is 8.

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
Steps:
Shawn started with 5 toys.
If he got 2 toys each from his mom and dad, then that is 4 more toys.
5 + 4 = 9.
The answer is 9.

Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
Steps:
There were originally 9 computers.
For each of 4 days, 5 more computers were added.
So 5 * 4 = 20 computers were added.
9 + 20 is 29.
The answer is 29.

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
Steps:
Michael started with 58 golf balls.
After losing 23 on tuesday, he had 58 - 23 = 35.
After losing 2 more, he had 35 - 2 = 33.
The answer is 33.

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
Steps:
Olivia had 23 dollars.
5 bagels for 3 dollars each will be 5 x 3 = 15 dollars.
So she has 23 - 15 dollars left.
23 - 15 is 8.
The answer is 8.

Q: {input}
'''
