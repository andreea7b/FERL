import numpy as np
import math

from scipy.optimize import minimize, newton, BFGS, NonlinearConstraint, Bounds, SR1, differential_evolution
from scipy.stats import chi2

class PHRILearner(object):
	"""
	This class performs correction inference given a trajectory and an input
	torque applied onto the trajectory.
	"""

	def __init__(self, feat_method, environment, constants):

		# ---- Important internal variables ---- #
		self.feat_method = feat_method
		self.environment = environment

		self.alpha = constants["alpha"]
		self.n = constants["n"]
		self.step_size = constants["step_size"]
		self.P_beta = constants["P_beta"]

	def learn_betas(self, traj, u_h, t, feat_idx = None):
		"""
		To estimate beta, we need to set up the optimization problem, compute
		optimal u_H wrt every feature, and calculate beta value per feature.
		---

		Params:
			traj [Trajectory] -- Current trajectory that force was applied to.
			u_h [array] -- Human force applied onto the trajectory.
			t [float] -- Time where deformation was applied.
			feat_idx [list] -- Optional parameter for feature index that needs beta.

		Returns:
			betas [list] -- a vector of beta values per feature.
		"""
		# if no list of idx is provided use all of them
		if feat_idx is None:
			feat_idx = list(np.arange(self.environment.num_features))

		betas = []
		traj_deform = traj.deform(u_h, t, self.alpha, self.n)
		new_features = self.environment.featurize(traj_deform.waypts)
		Phi_p = np.array([sum(x) for x in new_features])

		# Update betas vector.
		for i in feat_idx:
			# Set up the optimization problem.
			def u_constrained(u):
				cost = np.linalg.norm(u)**2
				return cost

			def u_jacobian(u):
				return 2*np.transpose(u)

			def u_hessian(u):	
				return 2 * np.eye(u.shape[0])

			def u_constraint(u):
				u_p = np.reshape(u, (7,1))
				waypts_deform_p = traj.deform(u_p, t, self.alpha, self.n).waypts
				H_features = self.environment.featurize(waypts_deform_p, [i])[0]
				Phi_H = sum(H_features)
				cost = (Phi_H - Phi_p[i])**2
				return cost
			
			def u_unconstrained(u):
				lambda_u = 5000
				u_p = np.reshape(u, (7,1))
				waypts_deform_p = traj.deform(u_p, t, self.alpha, self.n).waypts
				H_features = self.environment.featurize(waypts_deform_p, [i])[0]
				Phi_H = sum(H_features)
				cost = np.linalg.norm(u)**2 + lambda_u * (Phi_H - Phi_p[i])**2
				return cost

			# Compute optimal action.
			nonlinear_constraint = NonlinearConstraint(u_constraint, 0, 0, jac='2-point', hess=BFGS())
			#u_h_opt = minimize(u_constrained, np.squeeze(u_h), method='trust-constr',
			#				jac=u_jacobian, hess=u_hessian,
			#				constraints=[nonlinear_constraint], options={'xtol': 0, 'gtol': 0,'verbose': 2})
			u_h_opt = minimize(u_unconstrained, np.squeeze(u_h), method='L-BFGS-B', jac=u_jacobian)
			#u_h_opt = minimize(u_constrained, np.zeros((7,1)), method='SLSQP',
			#					constraints=({'type': 'eq', 'fun': u_constraint, 'args': (i,)}),
			#					options={'maxiter': 10, 'ftol': 1e-6, 'disp': True})
			#u_h_opt = differential_evolution(u_constrained, np.zeros((7,1)), constraints=(nonlinear_constraint))
			u_h_star = np.reshape(u_h_opt.x, (7, 1))

			waypts_deform_p = traj.deform(u_h_star, t, self.alpha, self.n).waypts
			H_features = self.environment.featurize(waypts_deform_p)
			Phi_u_star = np.array([sum(x) for x in H_features])
			print "Phi_p: ", Phi_p[i]
			print "Phi_p_H: ", Phi_u_star

			print "u_h: ", u_h, np.linalg.norm(u_h)
			print "u_h_star: ", u_h_star, np.linalg.norm(u_h_star)

			# Compute beta based on deviation from optimal action.
			beta_norm = 1.0 / np.linalg.norm(u_h_star) ** 2
			beta = self.environment.num_features / (2 * beta_norm * abs(np.linalg.norm(u_h)**2 - np.linalg.norm(u_h_star)**2))
			betas.append(beta)

		print "Here is beta:", betas
		return betas

	def learn_weights(self, traj, u_h, t, betas):
		"""
		Deforms the trajectory given human force, u_h, and
		updates features by computing difference between
		features of new trajectory and old trajectory.
		---

		Params:
			traj [Trajectory] -- Current trajectory that force was applied to.
			u_h [array] -- Human force applied onto the trajectory.
			t [float] -- Time where deformation was applied.
			betas [list] -- A list of the confidences to be used in learning.
		"""
		traj_deform = traj.deform(u_h, t, self.alpha, self.n)
		new_features = self.environment.featurize(traj_deform.waypts)
		old_features = self.environment.featurize(traj.waypts)
		Phi_p = np.array([sum(x) for x in new_features])
		Phi = np.array([sum(x) for x in old_features])
		update = Phi_p - Phi

		if self.feat_method == "all":
			# Update all weights. 
			curr_weight = self.all_update(update)
		elif self.feat_method == "max":
			# Update only weight of maximal change.
			curr_weight = self.max_update(update)
		elif self.feat_method == "beta":
			# Confidence matters. Update weights with it.
			curr_weight = self.confidence_update(update, betas)
		else:
			raise Exception('Learning method {} not implemented.'.format(self.feat_method))

		print "Here is the update:", update
		print "Here are the old weights:", self.environment.weights
		print "Here are the new weights:", curr_weight
		self.environment.weights = np.maximum(curr_weight, np.zeros(curr_weight.shape))

	def all_update(self, update):
		return self.environment.weights - self.step_size * update

	def max_update(self, update):
		# Get index of maximal change.
		max_idx = np.argmax(np.fabs(update))

		# Update only weight of feature with maximal change.
		curr_weight = np.array([self.environment.weights[i] for i in range(self.environment.num_features)])
		curr_weight[max_idx] = curr_weight[max_idx] - self.step_size*update[max_idx]
		return curr_weight

	def confidence_update(self, update, betas):
		"""
		To estimate theta, we need to retrieve the appropriate P(E | beta),
		then optimize the gradient step with Newton Rapson.
		---

		Params:
			update -- the full update step that will get weighted by confidence
			betas -- a vector of beta values for all features
		Returns:
			weights -- a vector of the new weights
		"""
		confidence = [1.0] * self.environment.num_features
		for i in range(self.environment.num_features):
			### Compute update using P(r|beta) for the beta estimate we just computed ###
			# Compute P(r|beta)
			mus1 = self.P_beta[self.environment.feature_list[i]+"1"]
			mus0 = self.P_beta[self.environment.feature_list[i]+"0"]
			p_r0 = chi2.pdf(betas[i],mus0[0],mus0[1],mus0[2]) / (chi2.pdf(betas[i],mus0[0],mus0[1],mus0[2]) + chi2.pdf(betas[i],mus1[0],mus1[1],mus1[2]))
			p_r1 = chi2.pdf(betas[i],mus1[0],mus1[1],mus1[2]) / (chi2.pdf(betas[i],mus0[0],mus0[1],mus0[2]) + chi2.pdf(betas[i],mus1[0],mus1[1],mus1[2]))
			l = math.pi

			# Newton-Rapson setup; define function, derivative, and call optimization method.
			def f_theta(weights_p):
				num = p_r1 * np.exp(weights_p * update[i])
				denom = p_r0 * (l/math.pi) ** (self.environment.num_features/2.0) * np.exp(-l*update[i]**2) + num
				return weights_p + self.step_size * num * update[i]/denom - self.environment.weights[i]
			def df_theta(weights_p):
				num = p_r0 * (l/math.pi) ** (self.environment.num_features/2.0) * np.exp(-l*update[i]**2)
				denom = p_r1 * np.exp(weights_p*update[i])
				return 1 + self.step_size * num / denom

			weight_p = newton(f_theta,self.environment.weights[i],df_theta,tol=1e-04,maxiter=1000)

			num = p_r1 * np.exp(weight_p * update[i])
			denom = p_r0 * (l/math.pi) ** (self.environment.num_features/2.0) * np.exp(-l*update[i]**2) + num
			confidence[i] = num/denom

		print "Here is weighted beta:", confidence
		weights = self.environment.weights - np.array(confidence) * self.step_size * update
		return weights

