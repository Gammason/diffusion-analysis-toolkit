# Diffusion Analysis Toolkit

## Overview

This repository contains a Python-based graphical tool for analyzing diffusion processes in thin-film systems using temperature-dependent experimental data.

The application calculates diffusion coefficients and diffusion lengths using Arrhenius relations and allows comparison of different transport mechanisms, including lattice diffusion, grain-boundary diffusion, surface diffusion, and effective combined diffusion.

## Key Features

* Graphical user interface (Tkinter-based)
* Batch processing of multiple CSV files
* Calculation of:

  * Diffusion coefficient D(T)
  * Diffusion length L(T)
* Support for:

  * Cr in Ni
  * Ni in Cr
* Adjustable physical parameters:

  * Activation energy (Q)
  * Pre-exponential factor (D₀)
  * Grain size and boundary width
  * Surface contribution
  * Film thickness
* Flexible plotting:

  * Temperature (K)
  * 1000/T
  * Pulse amplitude (mA)
* Log-scale visualization
* Multi-point probing and annotation
* Export of processed data and figures

* ## Interface Preview

The graphical interface allows interactive exploration of diffusion mechanisms, parameter tuning, and real-time visualization of diffusion coefficients and lengths.

![Diffusion GUI](figures/diffusion_gui_overview.png)

## Physical Model

Diffusion is calculated using the Arrhenius relation:

D(T) = D₀ · exp(-Q / RT)

The diffusion length is computed as:

L = √(2Dt)

Effective diffusion includes contributions from lattice, grain-boundary, and surface mechanisms weighted by geometric factors.

## Requirements

* Python 3.x
* numpy
* pandas
* matplotlib

Install dependencies:

pip install -r requirements.txt

## Usage

Run the application:

python diffusion_gui.py

Then:

1. Load a folder with CSV files
2. Select temperature and current columns
3. Set physical parameters
4. Run calculations
5. Visualize diffusion behavior

## Example Data

You can include a sample dataset in:

example_data/sample_temperature_data.csv

## Author

Elijah
Experimental physicist working on electromigration and diffusion in thin-film systems.
