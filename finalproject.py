# CSCI445 Final Project
# Participant: 
#   He Huang
#   Zixin Ding

from pyCreate2 import create2
import math
import odometry
import pid_controller
import lab8_map
import rrt_map
import particle_filter
import rrt
import numpy as np
import time



class Run:
    def __init__(self, factory):
        """Constructor.
        Args:
            factory (factory.FactoryCreate)
        """
        self.create = factory.create_create()
        self.time = factory.create_time_helper()
        self.servo = factory.create_servo()
        self.sonar = factory.create_sonar()
        self.arm = factory.create_kuka_lbr4p()
        self.virtual_create = factory.create_virtual_create()
        # self.virtual_create = factory.create_virtual_create("192.168.1.XXX")
        self.odometry = odometry.Odometry()
        self.mapJ = lab8_map.Map("lab8_map.json")
        self.map = rrt_map.Map("configuration_space.png")
        self.rrt = rrt.RRT(self.map)

        # TODO identify good PID controller gains
        self.pidTheta = pid_controller.PIDController(300, 5, 50, [-10, 10], [-200, 200], is_angle=True)#pid_controller.PIDController(200, 0, 100, [-10, 10], [-50, 50], is_angle=True)
        # TODO identify good particle filter parameters
        self.pf = particle_filter.ParticleFilter(self.mapJ, 1200, 0.01, 0.05, 0.1)#particle_filter.ParticleFilter(self.mapJ, 1000, 0.06, 0.15, 0.2)

        self.joint_angles = np.zeros(7)

        # goal location
        self.goal_position = (1.6, 2.5)
        self.goal_shelf = 1

        self.offset3 = 0.3105
        #self.arm_x = 1.6001
        #self.arm_y = 3.3999
        self.l1 = 0.4
        self.l2 = 0.39

    def sleep(self, time_in_sec):
        """Sleeps for the specified amount of time while keeping odometry up-to-date
        Args:
            time_in_sec (float): time to sleep in seconds
        """
        start = self.time.time()
        while True:
            state = self.create.update()
            if state is not None:
                self.odometry.update(state.leftEncoderCounts, state.rightEncoderCounts)
                # print("[{},{},{}]".format(self.odometry.x, self.odometry.y, math.degrees(self.odometry.theta)))
            t = self.time.time()
            if start + time_in_sec <= t:
                break

    def go_to_angle(self, goal_theta):
        old_x = self.odometry.x
        old_y = self.odometry.y
        old_theta = self.odometry.theta
        while math.fabs(math.atan2(
            math.sin(goal_theta - self.odometry.theta),
            math.cos(goal_theta - self.odometry.theta))) > 0.02:
            output_theta = self.pidTheta.update(self.odometry.theta, goal_theta, self.time.time())
            self.create.drive_direct(int(+output_theta)+5, int(-output_theta)-5)
            self.sleep(0.01)
        self.create.drive_direct(0, 0)
        self.pf.move_by(self.odometry.x - old_x, self.odometry.y - old_y, self.odometry.theta - old_theta)

    def forward(self):
        old_x = self.odometry.x
        old_y = self.odometry.y
        old_theta = self.odometry.theta
        base_speed = 100
        distance = 0.5
        goal_x = self.odometry.x + math.cos(self.odometry.theta) * distance
        goal_y = self.odometry.y + math.sin(self.odometry.theta) * distance
        while True:
            goal_theta = math.atan2(goal_y - self.odometry.y, goal_x - self.odometry.x)
            output_theta = self.pidTheta.update(self.odometry.theta, goal_theta, self.time.time())
            self.create.drive_direct(int(base_speed+output_theta), int(base_speed-output_theta))

            # stop if close enough to goal
            distance = math.sqrt(math.pow(goal_x - self.odometry.x, 2) + math.pow(goal_y - self.odometry.y, 2))
            if distance < 0.05:
                #self.create.drive_direct(0, 0)
                break
            self.sleep(0.01)
        self.pf.move_by(self.odometry.x - old_x, self.odometry.y - old_y, self.odometry.theta - old_theta)

    def visualize(self):
        x, y, theta = self.pf.get_estimate()
        self.virtual_create.set_pose((x, y, 0.1), theta)
        data = []
        #data.extend([self.odometry.x, self.odometry.y, 0.1, self.odometry.theta])
        for particle in self.pf._particles:
            data.extend([particle.x, particle.y, 0.1, particle.theta])
        self.virtual_create.set_point_cloud(data)

    def real_to_image(self, position):
        image_x = position[0] * 100
        image_y = self.map.height - position[1]*100
        return (int(image_x), int(image_y))

    def distance(self, a, b):
        return math.sqrt(math.pow(a[0] - b[0], 2) + math.pow(a[1] - b[1], 2))

    def image_to_real(self, position):
        real_x = position[0] / 100.
        real_y = (self.map.height - position[1]) / 100.
        return (real_x, real_y)
    
    def final_rotate_angle(self):
        self.arm.get_position()
        self.time.sleep(1)
        arm_location = self.arm.get_position()

        if arm_location[1] > 3:
            return -math.pi/2.
        elif arm_location[1] < 0:
            return math.pi/2.
        elif arm_location[0] < 0:
            return 0
        elif arm_location[0] > 3:
            return math.pi


    ### section2
    def inverse_kinematics(self, _x, _z):
        z = _z - self.offset3
        x = _x
        r = math.sqrt(x**2 + z**2)
        alpha = math.acos((self.l1**2 + self.l2**2 - r**2)/(2*self.l1*self.l2))
        theta2 = math.pi - alpha
        beta = math.acos((r**2 + self.l1**2 - self.l2**2)/(2*self.l1*r))
        theta1 = math.atan2(x, z) - beta

        if theta2 < -math.pi/2 or theta2 > math.pi/2 or theta1 < -math.pi/2 or theta1 > math.pi/2:
            theta2 = math.pi + alpha
            theta1 = math.atan2(x, z) + beta
        theta3 = math.pi/2 - (theta1 + theta2)
        self.arm.go_to(5, theta3)
        self.arm.go_to(1, theta1)
        self.arm.go_to(3, theta2)
        #print("Go to [{},{}], IK: [{} deg, {} deg]".format(x, z, math.degrees(theta1), math.degrees(theta2)))

    def grip_and_place(self, dist, shelf):
        _z = 0.125
        self.inverse_kinematics(dist, 0.3)
        for z in np.arange(0.3, _z, -0.01):
            self.inverse_kinematics(dist, z)
            self.time.sleep(0.05)
        self.time.sleep(5)
        #self.inverse_kinematics(2.5, 0.1617)
        self.arm.close_gripper()
        self.time.sleep(5)
        # Placing cup onto shelf, constants are based on relative postion of arm and shelf
        for z in np.arange(_z, 0.30, 0.01):
            self.inverse_kinematics(dist, z)
            self.time.sleep(0.05)
        if shelf == 1:
            self.inverse_kinematics(0.5399, 0.55)
            self.time.sleep(5)
            self.arm.go_to(0, -3*math.pi/4)
        elif shelf == 2:
            self.inverse_kinematics(0.36, 0.85)
            self.time.sleep(5)
            self.arm.go_to(0, -4*math.pi/5)
        elif shelf == 3:
            self.arm.go_to(3, 0)
            self.arm.go_to(5, -math.pi/6)
            self.time.sleep(5)
            self.arm.go_to(1, -math.pi/6)
        self.time.sleep(5)
        self.arm.open_gripper()
        self.time.sleep(10)

    # # returns closest wall, and distance betweeen arm and cup
    def distance_to_arm(self, x, y):
        wall_dist = (0, 0)
        min = 3.025
        dist_to_walls = [(1,math.fabs(x - 0.025)), (2, math.fabs(y - 0.025)), (3, math.fabs(x - 3.025)), (4, math.fabs(y - 3.025))]
        for pair in dist_to_walls:
            if pair[1] < min:
                min = pair[1]
                wall_dist = (pair[0], pair[1] + 0.3749 - 0.26 - 0.058) # -0.058
        print("Closest to wall ", wall_dist[0], "; x distance: ", wall_dist[1])
        return wall_dist

    def rotate_link0(self, wall_dist, location):
        if wall_dist[0] == 1:
            self.arm.go_to(0, (location[1] - self.goal_position[1]))
            self.time.sleep(5)
        elif wall_dist[0] == 2:
            self.arm.go_to(0, (self.goal_position[0] - location[0]))
            self.time.sleep(5)
        elif wall_dist[0] == 3:
            self.arm.go_to(0, (self.goal_position[1] - location[1]))
            self.time.sleep(5)
        elif wall_dist[0] == 4:
            self.arm.go_to(0, (location[0] - self.goal_position[0]))
            self.time.sleep(5)

    def get_position(self):
        self.create.sim_get_position()
        self.time.sleep(1)
        position = self.create.sim_get_position()
        return position

    def compensate_location(self):
        current_loc = self.get_position()
        self.odometry.x = current_loc[0]
        self.odometry.y = current_loc[1]
        base_speed = 15
        while True:
            state = self.create.update()
            old_x = self.odometry.x
            old_y = self.odometry.y
            old_theta = self.odometry.theta
            if state is not None:
                self.odometry.update(state.leftEncoderCounts, state.rightEncoderCounts)
                goal_theta = math.atan2(self.goal_position[1] - self.odometry.y, self.goal_position[0] - self.odometry.x)
                theta = math.atan2(math.sin(self.odometry.theta), math.cos(self.odometry.theta))
                output_theta = self.pidTheta.update(self.odometry.theta, goal_theta, self.time.time())
                self.create.drive_direct(int(base_speed+output_theta), int(base_speed-output_theta))

                distance = math.sqrt(math.pow(self.goal_position[0] - self.odometry.x, 2) + math.pow(self.goal_position[1] - self.odometry.y, 2))
                x_distance = abs(self.odometry.x - self.goal_position[0])
                if x_distance < 0.017 and distance < 0.04:
                    self.create.drive_direct(0, 0)
                    break

    def run(self):
        start_time = time.time()
        self.create.start()
        self.create.safe()

        self.create.drive_direct(0, 0)
        # build the map
        self.create.sim_get_position()
        self.time.sleep(2)
        start_position = self.create.sim_get_position()

        self.odometry.x = start_position[0]
        self.odometry.y = start_position[1]
        self.odometry.theta = 0
        self.pf.set_particle((start_position[0], start_position[1], self.odometry.theta))
        start_position = self.real_to_image(start_position)
        self.rrt.build(start_position, 6000, 6)
        goal = self.rrt.nearest_neighbor(self.real_to_image(self.goal_position))
        path = self.rrt.shortest_path(goal)
        base_speed = 100

        # this is for visualize path
        for v in self.rrt.T:
            for u in v.neighbors:
                self.map.draw_line((v.state[0], v.state[1]), (u.state[0], u.state[1]), (0,0,0))
        for idx in range(0, len(path)-1):
            self.map.draw_line((path[idx].state[0], path[idx].state[1]), (path[idx+1].state[0], path[idx+1].state[1]), (0,255,0))
        self.map.save("lab10_rrt.png")

        # request sensors
        self.create.start_stream([
            create2.Sensor.LeftEncoderCounts,
            create2.Sensor.RightEncoderCounts,
        ])
        self.visualize()
        self.virtual_create.enable_buttons()
        self.visualize()

        count = 0
        force_sense = False

        # make the init rotation
        initial_path = self.image_to_real(path[0].state)
        second_path = self.image_to_real(path[1].state)
        goal_theta = math.atan2(second_path[1] - initial_path[1], second_path[0] - initial_path[0])
        self.go_to_angle(goal_theta)

        # execute the path
        for p in path:
            # avoid initial position
            if count == 0:
                count += 1
                distance = self.sonar.get_distance()
                self.pf.measure(distance, 0)
                continue
            real_position = self.image_to_real(p.state)
            goal_x = real_position[0]
            goal_y = real_position[1]
            while True:
                state = self.create.update()
                old_x = self.odometry.x
                old_y = self.odometry.y
                old_theta = self.odometry.theta
                if state is not None:
                    self.odometry.update(state.leftEncoderCounts, state.rightEncoderCounts)
                    goal_theta = math.atan2(goal_y - self.odometry.y, goal_x - self.odometry.x)
                    theta = math.atan2(math.sin(self.odometry.theta), math.cos(self.odometry.theta))
                    output_theta = self.pidTheta.update(self.odometry.theta, goal_theta, self.time.time())
                    self.create.drive_direct(int(base_speed+output_theta), int(base_speed-output_theta))

                    distance = math.sqrt(math.pow(goal_x - self.odometry.x, 2) + math.pow(goal_y - self.odometry.y, 2))
                    if distance < 0.05:
                        self.create.drive_direct(0, 0)
                        break
                    self.pf.move_by(self.odometry.x - old_x, self.odometry.y - old_y, self.odometry.theta - old_theta)
            if force_sense or count % 3 == 0 or count == len(path) - 1:
                distance = self.sonar.get_distance()
                distance_on_map = self.mapJ.closest_distance([self.odometry.x, self.odometry.y], self.odometry.theta)
                #print("Sonar sense: {}, {}".format(distance, distance_on_map))
                if distance_on_map < 0.12 or distance_on_map > 3.3:
                    force_sense = True
                    continue
                self.pf.measure(distance, 0)
                force_sense = False

                x, y, theta = self.pf.get_estimate()
                error_pos = self.distance((x, y), self.get_position()) #self.get_position()   (self.odometry.x, self.odometry.y)
                error_theta = abs(theta - self.odometry.theta)
                #print("{}: {}".format(error_pos, error_theta))
                if error_pos < 0.05:
                    self.odometry.x = x
                    self.odometry.y = y
                if error_theta < 0.02:
                    self.odometry.theta = theta
                #print(math.degrees(theta))
                #print("Error: {}, {}, Estimate pos: ({}, {}, {}), odometry pos: ({}, {}, {})".format(error_pos, error_theta, x, y, theta, self.odometry.x, self.odometry.y, self.odometry.theta))
            self.visualize()
            count += 1

        self.create.drive_direct(0, 0)
        print("Final Step")
        self.compensate_location()
        self.go_to_angle(self.final_rotate_angle())
        self.create.drive_direct(0, 0)
        reached_goal_location = (self.odometry.x, self.odometry.y)
        #print("odometry theta: {}".format(math.degrees(self.odometry.theta)))
        #print("estimate goal location: {}".format(reached_goal_location))
        location = self.get_position()
        print("goal location: {}".format(location))
        print("error: {}".format(self.distance(reached_goal_location, location)))
        #location = reached_goal_location


        # distance_to_arm needs cup's absolute x, y coordinates
        # for current arm and shelf setup, cup's x, y need to satisfy x = 1.6, y = 2.38~2.59
        wall_dist = self.distance_to_arm(location[0], location[1])
        self.rotate_link0(wall_dist, location)

        # self.grip_and_place(wall_dist[1], 1)
        self.grip_and_place(wall_dist[1], self.goal_shelf)
        # self.grip_and_place(wall_dist[1], 3)

        print("--- %s seconds ---" % (time.time() - start_time))
        while True:
            shit = 0

        '''
        while True:
            b = self.virtual_create.get_last_button()
            if b == self.virtual_create.Button.MoveForward:
                self.forward()
                self.visualize()
            elif b == self.virtual_create.Button.TurnLeft:
                self.go_to_angle(self.odometry.theta + math.pi / 2)
                self.visualize()
            elif b == self.virtual_create.Button.TurnRight:
                self.go_to_angle(self.odometry.theta - math.pi / 2)
                self.visualize()
            elif b == self.virtual_create.Button.Sense:
                distance = self.sonar.get_distance()
                #print(distance)
                self.pf.measure(distance, 0)
                self.visualize()


            self.time.sleep(0.01)
        '''
