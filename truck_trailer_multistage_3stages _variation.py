#
#     This file is part of rockit.
#
#     rockit -- Rapid Optimal Control Kit
#     Copyright (C) 2019 MECO, KU Leuven. All rights reserved.
#
#     Rockit is free software; you can redistribute it and/or
#     modify it under the terms of the GNU Lesser General Public
#     License as published by the Free Software Foundation; either
#     version 3 of the License, or (at your option) any later version.
#
#     Rockit is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#     Lesser General Public License for more details.
#
#     You should have received a copy of the GNU Lesser General Public
#     License along with CasADi; if not, write to the Free Software
#     Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
#
#

"""
Motion planning
===============

Simple motion planning for vehicle with trailer
"""

from rockit import *
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, cos, sin, tan
from casadi import vertcat, horzcat
from plot_trailer import *
from simulator import *
import yaml
import casadi as ca
from casadi import DM, evalf
import csv

show_figures = True
show_gridpoints = False
use_simulator = False
save_for_gif = False

refine = 2
Ts = 0.1

# Environment
# position, orientation and width corridor 1
pc_co_1 = vertcat(0., 2.5)
or_co_1 = -90/180*pi
wi_co_1 = 1.
le_co_1 = 5.

# position, orientation and width corridor 2
pc_co_2 = vertcat(0., 2.8)
or_co_2 = 240/180*pi
wi_co_2 = .5
le_co_2 = 5.

# vehicle orientation at start and finish
veh_orient_1 = or_co_1 + pi
veh_orient_2 = or_co_2


def get_intersection(w1, w2):
    return np.array([(w1[1]*w2[2] - w1[2]*w2[1])/(w1[0]*w2[1] - w1[1]*w2[0]),
	                 (w1[2]*w2[0] - w1[0]*w2[2])/(w1[0]*w2[1] - w1[1]*w2[0])])



def define_corridor(or_co, wi_co, le_co, pc_co):
	si_co = sin(or_co)
	co_co = cos(or_co)

	n1 = vertcat(co_co, si_co)
	t1 = pc_co + le_co*n1
	w1 = vertcat(n1, -n1.T @ t1)

	n2 = vertcat(-si_co, co_co)
	t2 = pc_co + wi_co/2*n2
	w2 = vertcat(n2, -n2.T @ t2)

	n3 = -n1
	t3 = pc_co + wi_co/2*n3
	w3 = vertcat(n3, -n3.T @ t3)

	n4 = -n2
	t4 = pc_co + wi_co/2*n4
	w4 = vertcat(n4, -n4.T @ t4)

	return w1, w2, w3, w4


w1_1, w1_2, w1_3, w1_4 = define_corridor(or_co_1, wi_co_1, le_co_1, pc_co_1)
w2_1, w2_2, w2_3, w2_4 = define_corridor(or_co_2, wi_co_2, le_co_2, pc_co_2)

room1 = horzcat(w1_1, w1_2, w1_3, w1_4)
room2 = horzcat(w2_1, w2_2, w2_3, w2_4)

corner1_1 = get_intersection(w1_1, w1_2)
corner1_2 = get_intersection(w1_2, w1_3)
corner1_3 = get_intersection(w1_3, w1_4)
corner1_4 = get_intersection(w1_4, w1_1)

corner2_1 = get_intersection(w2_1, w2_2)
corner2_2 = get_intersection(w2_2, w2_3)
corner2_3 = get_intersection(w2_3, w2_4)
corner2_4 = get_intersection(w2_4, w2_1)


for x_start in np.linspace(-0.3, 0.3,100):
	for y_start in np.linspace(-1, 1,100):
		x1_t0 = (sum([corner1_1[0], corner1_2[0], corner1_3[0], corner1_4[0]]) / 4)+x_start+(0.6/99)*0.5
		y1_t0 = (sum([corner1_1[1], corner1_2[1], corner1_3[1], corner1_4[1]]) / 4)+y_start+(2/99)*0.5
		theta1_t0 = veh_orient_1
		theta0_t0 = veh_orient_1

		x1_tf = sum([corner2_1[0], corner2_2[0], corner2_3[0], corner2_4[0]]) / 4
		y1_tf = sum([corner2_1[1], corner2_2[1], corner2_3[1], corner2_4[1]]) / 4
		theta1_tf = veh_orient_2
		theta0_tf = veh_orient_2

		# Parameters
		with open('truck_trailer_para.yaml', 'r') as file:
			para = yaml.safe_load(file)

		L0 = para['truck']['L']
		M0 = para['truck']['M']
		W0 = para['truck']['W']
		L1 = para['trailer1']['L']
		M1 = para['trailer1']['M']
		W1 = para['trailer1']['W']


		def create_stage(ocp, t0, T, N, M, truck, trailer):
			stage = ocp.stage(t0=t0, T=T)

			# Trailer model
			theta1 = stage.state()
			x1     = stage.state()
			y1     = stage.state()

			theta0 = stage.state()
			x0     = x1 + L1*cos(theta1) + M0*cos(theta0)
			y0     = y1 + L1*sin(theta1) + M0*sin(theta0)

			#delta0 = stage.control(order=1)
			v0      = stage.control(order=1)
			dtheta0 = stage.control(order=1)  # v0/L0*tan(delta0)

			beta01 = theta0 - theta1

			dtheta1 = v0/L1*sin(beta01) - M0/L1*cos(beta01)*dtheta0
			v1 = v0*cos(beta01) + M0*sin(beta01)*dtheta0

			stage.set_der(theta1, dtheta1)
			stage.set_der(x1,     v1*cos(theta1))
			stage.set_der(y1,     v1*sin(theta1))

			stage.set_der(theta0, dtheta0)

			# Path constraints
			stage.subject_to(-.2 <= (v0 <= .2))
			stage.subject_to(-1 <= (stage.der(v0) <= 1))

			#stage.subject_to(-pi/6 <= (delta0 <= pi/6))
			#stage.subject_to(-pi/10 <= (stage.der(delta0) <= pi/10))
			stage.subject_to(-pi/10 <= (stage.der(dtheta0) <= pi/10))

			stage.subject_to(-pi/4 <= (beta01 <= pi/4))

			# Room constraint
			veh_vertices = get_vehicle_vertices(x0, y0, theta0, W0/2, W0/2, L0, M0)
			for i in range(veh_vertices.size(2)):
				veh_vertex = veh_vertices[:, i]
				phom = vertcat(veh_vertex[0], veh_vertex[1], 1)
				stage.subject_to(truck.T @ phom <= 0, grid='integrator')

			veh_vertices = get_vehicle_vertices(x1, y1, theta1, W1/2, W1/2, L1, M1)
			for i in range(veh_vertices.size(2)):
				veh_vertex = veh_vertices[:, i]
				phom = vertcat(veh_vertex[0], veh_vertex[1], 1)
				stage.subject_to(trailer.T @ phom <= 0, grid='integrator')

			stage.method(MultipleShooting(N=N, M=M, intg='rk'))

			# Minimal time
			stage.add_objective(stage.T)
			# stage.add_objective(stage.integral(5*((x1-x1_tf)**2+(y1-y1_tf)**2)))
			# stage.add_objective(stage.integral(0.3*((theta1-theta1_tf)**2+(theta0-theta0_tf)**2)))
			stage.add_objective(stage.integral(dtheta0**2))

			return stage, theta1, x1, y1, theta0, x0, y0, v0, dtheta0


		def stitch_stages(ocp, stage1, stage2, time1, time2):
			# Stitch time
			ocp.subject_to(time1 == stage2.t0)
			# Stitch states
			for i in range(len(stage1.states)):
				ocp.subject_to(stage2.at_t0(stage2.states[i]) == stage1.at_tf(stage1.states[i]))

		T_end=30
		ocp = Ocp()

		# Stage 1 - Approach
		N_1 = 20
		M_1 = 2
		T_1 = 20.
		N_2 = 5
		M_2 = 2
		T_2 = 5.
		N_3 = 10
		M_3 = 2
		T_3 = 10.

		stage_1, theta1_1, x1_1, y1_1, theta0_1, x0_1, y0_1, v0_1, dtheta0_1 = \
			create_stage(ocp, 0, T_1, N_1, M_1,
						truck=room1, trailer=room1)

		# Initial constraints
		ocp.subject_to(stage_1.t0 == 0)
		ocp.subject_to(stage_1.at_t0(x1_1) == x1_t0)
		ocp.subject_to(stage_1.at_t0(y1_1) == y1_t0)
		ocp.subject_to(stage_1.at_t0(theta1_1) == theta1_t0)
		ocp.subject_to(stage_1.at_t0(theta0_1) == theta0_t0)

		# Stage 2 - Corner
		stage_2, theta1_2, x1_2, y1_2, theta0_2, x0_2, y0_2, v0_2, dtheta0_2 = \
				create_stage(ocp, T_1, T_2, N_2, M_2, \
							truck=room2, trailer=room1)
		stitch_stages(ocp, stage_1, stage_2, T_1, T_2)

		# Stage 3 - Exit
		stage_3, theta1_3, x1_3, y1_3, theta0_3, x0_3, y0_3, v0_3, dtheta0_3 = \
				create_stage(ocp, T_1+T_2, T_3, N_3, M_3, \
							truck=room2, trailer=room2)
		stitch_stages(ocp, stage_2, stage_3, T_1+T_2, T_3)

		# Final constraint
		ocp.subject_to(stage_3.at_tf(x1_3) == x1_tf)
		ocp.subject_to(stage_3.at_tf(y1_3) == y1_tf)
		ocp.subject_to(stage_3.at_tf(theta1_3) == theta1_tf)
		ocp.subject_to(stage_3.at_tf(theta0_3) == theta0_tf)

		# Pick a solution method
		options = { "expand": True,
					"verbose": False,
					"print_time": False,
					"error_on_fail": False,
					"ipopt": {	#"linear_solver": "ma57",
								"print_level": 0,
								"tol": 1e-8}}
		ocp.solver('ipopt', options)


		theta1_1s = stage_1.sample(theta1_1, grid='control')[1]
		x1_1s     = stage_1.sample(x1_1, 	 grid='control')[1]
		y1_1s     = stage_1.sample(y1_1, 	 grid='control')[1]
		theta0_1s = stage_1.sample(theta0_1, grid='control')[1]
		# delta0_1s = stage_1.sample(delta0_1, grid='control')[1]
		v0_1s      = stage_1.sample(v0_1, 	   grid='control')[1]
		dtheta0_1s = stage_1.sample(dtheta0_1, grid='control')[1]

		theta1_2s = stage_2.sample(theta1_2, grid='control')[1]
		x1_2s     = stage_2.sample(x1_2, 	 grid='control')[1]
		y1_2s     = stage_2.sample(y1_2, 	 grid='control')[1]
		theta0_2s = stage_2.sample(theta0_2, grid='control')[1]
		# delta0_2s = stage_2.sample(delta0_2, grid='control')[1]
		v0_2s      = stage_2.sample(v0_2, 	   grid='control')[1]
		dtheta0_2s = stage_2.sample(dtheta0_2, grid='control')[1]

		theta1_3s = stage_3.sample(theta1_3, grid='control')[1]
		x1_3s     = stage_3.sample(x1_3, 	 grid='control')[1]
		y1_3s     = stage_3.sample(y1_3, 	 grid='control')[1]
		theta0_3s = stage_3.sample(theta0_3, grid='control')[1]
		# delta0_3s = stage_3.sample(delta0_3, grid='control')[1]
		v0_3s     = stage_3.sample(v0_3, 	 grid='control')[1]
		dtheta0_3s = stage_3.sample(dtheta0_3, grid='control')[1]

		sampler1  = stage_1.sampler([theta1_1, x1_1, y1_1, theta0_1, x0_1, y0_1, v0_1, dtheta0_1])
		sampler2  = stage_2.sampler([theta1_2, x1_2, y1_2, theta0_2, x0_2, y0_2, v0_2, dtheta0_2])
		sampler3  = stage_3.sampler([theta1_3, x1_3, y1_3, theta0_3, x0_3, y0_3, v0_3, dtheta0_3])

		t1 = ocp.value(stage_1.T)
		t2 = t1 + ocp.value(stage_2.T)
		t3 = t2 + ocp.value(stage_3.T)


		# Define solve_ocp function (ensure correct types for arguments)
		solve_ocp = ocp.to_function('solve_ocp',
									[theta1_1s, x1_1s, y1_1s, theta0_1s, v0_1s, dtheta0_1s,
									theta1_2s, x1_2s, y1_2s, theta0_2s, v0_2s, dtheta0_2s,
									theta1_3s, x1_3s, y1_3s, theta0_3s, v0_3s, dtheta0_3s],
									[t1, theta1_1s, x1_1s, y1_1s, theta0_1s, v0_1s, dtheta0_1s,
									t2, theta1_2s, x1_2s, y1_2s, theta0_2s, v0_2s, dtheta0_2s,
									t3, theta1_3s, x1_3s, y1_3s, theta0_3s, v0_3s, dtheta0_3s, ocp.gist])


		# Convert NumPy arrays to flat lists before passing to CasADi DM
		y1_array = DM(np.linspace(y1_t0, y1_tf, N_1+1).flatten().tolist())
		theta1_array = DM(np.linspace(theta1_t0, theta1_tf, N_2+1).flatten().tolist())
		theta0_array = DM(np.linspace(theta0_t0, theta0_tf, N_2+1).flatten().tolist())
		x1_array = DM(np.linspace(x1_t0, x1_tf, N_3+1).flatten().tolist())

		# Solve function
		t1_sol, theta1_1sol, x1_1sol, y1_1sol, theta0_1sol, v0_1sol, dtheta0_1sol, \
			t2_sol, theta1_2sol, x1_2sol, y1_2sol, theta0_2sol, v0_2sol, dtheta0_2sol, \
			t3_sol, theta1_3sol, x1_3sol, y1_3sol, theta0_3sol, v0_3sol, dtheta0_3sol, gist_sol = \
				solve_ocp(theta1_t0, x1_t0, y1_array, theta0_t0, .1, 0.,
						theta1_array, x1_t0, y1_tf, theta0_array, .1, 0.,
						theta1_tf, x1_array, y1_tf, theta0_tf, .1, 0.)



		theta1_ctrl = np.concatenate([theta1_1sol, theta1_2sol, theta1_3sol], axis=1)[0]
		x1_ctrl     = np.concatenate([x1_1sol, x1_2sol, x1_3sol], axis=1)[0]
		y1_ctrl     = np.concatenate([y1_1sol, y1_2sol, y1_3sol], axis=1)[0]
		theta0_ctrl = np.concatenate([theta0_1sol, theta0_2sol, theta0_3sol], axis=1)[0]
		# x0_ctrl     = np.concatenate([x0_1sol, x0_2sol, x0_3ctrl])
		# y0_ctrl     = np.concatenate([y0_1ctrl, y0_2ctrl, y0_3ctrl])
		# delta0_ctrl = np.concatenate([delta0_1ctrl, delta0_2ctrl, delta0_3ctrl])
		v0_ctrl      = np.concatenate([v0_1sol, v0_2sol, v0_3sol], axis=1)[0]
		dtheta0_ctrl = np.concatenate([dtheta0_1sol, dtheta0_2sol, dtheta0_3sol], axis=1)[0]
		t_ctrl       = np.concatenate([t1_sol, t2_sol, t3_sol], axis=1)[0]

		for idx, theta_var in enumerate(theta1_ctrl):
			with open('truck_trailer_multi_stage_correct_version_2.csv', "a+") as output_file:
				writer = csv.writer(output_file, lineterminator='\n')
				writer.writerow([x1_ctrl[idx], y1_ctrl[idx], theta0_ctrl[idx], theta1_ctrl[idx], v0_ctrl[idx], dtheta0_ctrl[idx]])


		# plt.figure(3)

		# ax1 = plt.subplot(1, 1, 1)
		# ax1.plot(x1_ctrl, y1_ctrl)

		# plt.show()