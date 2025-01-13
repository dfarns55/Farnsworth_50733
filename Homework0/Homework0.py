#!/usr/bin/env python3

"""
Homework0.py

     You've been given a file helpfully named `fakeData.txt`. Your mission,
     should you choose to accept it, is to:

    - Read in that file
    - Seperate it into two sets using the bad column (-1 means bad)
    - Plot the good data
    - Plot the bad data using Xs to clearly denote it
    - Fit a line to the data using an appropriate polynomial function

    You should accept the mission, it is a completion grade. The purpose of
    this assignment is to see how much you remember from your intro to
    programming class, and to see if you developed and very good or very bad
    habits. Please do your best to demonstrate good habits! (e.g., descriptive
    variable names, readable code, maybe even proper PEP 8 coding style!) You
    may use any resources at your disposal, except your friends please. I want
    to see what **YOU** remember. But your textbook, the internet, any lecture
    notes or homework you have from your previous all, are all free game as
    reference for this assignment. Just try to avoid copy-pasting too much
    (I'll be able to tell).


Author: Danette Farnsworth
Date: 2025-01-13
"""

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as sm


def main():
    """Read fakeData.txt, plot the good and bad, and add a polynimal fit"""

    # read the data
    fake_data = pd.read_csv("fakeData.txt", skipinitialspace=True)

    # separate into a good and bad fake_dataset
    # also sort, so it's easier to plot later
    bad_data = fake_data.loc[fake_data.bad == -1].sort_values("x")
    good_data = fake_data.loc[fake_data.bad != -1].sort_values("x")

    # plot the good fake_data, then the bad
    plt.scatter(good_data.x, good_data.y)
    plt.title("Good and Bad Data")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.scatter(bad_data.x, bad_data.y, marker="X")

    # find the right polynomial
    reg = {}
    reg[1] = sm.ols("y~x", data=good_data).fit()
    reg[2] = sm.ols("y~I(x**2)+x", data=good_data).fit()
    reg[3] = sm.ols("y~I(x**3)+I(x**2)+x", data=good_data).fit()
    reg[4] = sm.ols("y~I(x**4)+I(x**3)+I(x**2)+x", data=good_data).fit()
    reg[5] = sm.ols("y~I(x**5)+I(x**4)+I(x**3)+I(x**2)+x", data=good_data).fit()

    # according to reg_5.summary(), the highest order coefficient is not
    # statistically significant (p = 0.076) so I will use reg_4.

    # Also, reg_4 has the lowest bic
    bic = {x: reg[x].bic for x in reg}
    polynomial_order = min(bic, key=bic.get)
    best_model = reg[polynomial_order]
    print(f"The best model is a polynomial of order {polynomial_order}")

    # sort the data and plot the line of best fit
    plt.plot(good_data.x, best_model.predict(), color="red")

    # add a descriptive legend
    plt.legend(["Good", "Bad", f"Polynomial Approx ({polynomial_order}th order)"])
    plt.show()


if __name__ == "__main__":
    main()
