# Computer Science For Business Analytics Paper

This repository contains my code for the CSBA Paper: Scalable Product Duplicate Detection
The used data set contains 1624 tv's from four different webshops. 
By utilizing LSH with min-hashing a scalable method is created to detect duplicate products.
For details on the method and the results see the paper.

I wrote the code individually.

## File Structure
The main file is `main.py` which contains almost all the needed functions.
Going from top to bottom, the functions roughly follow the order of the method in the paper.

The `product.py` file is to create the product objects. 
When the objects are created most of the data cleaning is automatically done.

The `analyse_data.py` file is just for some basic analysis of the data. And plays no part in getting the final results.

The `for_loop.py` file is to run the main file multiple times with different parameters. 
I programmed this on demand, thus it is not very clean.

The `bootstrap.py` file is to run the main file with different bootstraps. 
I also added the plotting of the results for the paper in this part.

The `plot evaluation.py` file is made specifically for the plotting of the evaluation of my added methods. 
To be able to play around with the plot setting it uses a csv file that is created by the bootstrap file. 

