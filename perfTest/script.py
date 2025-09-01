import numpy as np

lower_limit = [robot.joints[0].lower_limit, robot.joints[1].lower_limit, robot.joints[2].lower_limit, robot.joints[3].lower_limit, robot.joints[4].lower_limit, robot.joints[5].lower_limit]
upper_limit = [robot.joints[0].upper_limit, robot.joints[1].upper_limit, robot.joints[2].upper_limit, robot.joints[3].upper_limit, robot.joints[4].upper_limit, robot.joints[5].upper_limit]

counter = 0
while counter < 99999:
    if (counter%2) == 0:
        display("A")
        target_trans = np.array([
       [ 0.11 , -0.239,  0.965,  0.797],
       [ 0.795,  0.603,  0.059,  0.165],
       [-0.596,  0.761,  0.257,  0.772],
       [ 0.   ,  0.   ,  0.   ,  1.   ]])
    else:
        display("B")
        target_trans = np.array([[-0.212,  0.676, -0.706, -0.625],
       [-0.633,  0.455,  0.626,  0.274],
       [ 0.744,  0.58 ,  0.331,  0.949],
       [ 0.   ,  0.   ,  0.   ,  1.   ]])

    #target_trans = robot.compute_global_transform()["tool0"]
    R = target_trans[:3, :3]
    P = target_trans[:3,  3]

    q0 = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

    q_sol = robot.dh.inverse_kinematics6D_with_limits(P, R, q0, 100, 0.0001)
    print("IK Lösung:", [int(np.rad2deg(x))%360 for x in q_sol])

    # FK prüfen
    for name, angle in zip(robot.dh.symbolic_thetas.keys(), q_sol):
        robot.dh.update_joint_angle(name, angle)
    fk_result = robot.dh.forward_kinematics()
    print("Erreichte Position:", fk_result[:3, 3])

    robot.animate_by_theta(q_sol, 4, 1, True)

    counter +=1