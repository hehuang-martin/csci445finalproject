from pyCreate2 import create2
import math
import odometry
import pid_controller
import lab8_map
import rrt_map
import particle_filter
import rrt
import numpy as np



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
        self.pf = particle_filter.ParticleFilter(self.mapJ, 1000, 0.01, 0.05, 0.1)#particle_filter.ParticleFilter(self.mapJ, 1000, 0.06, 0.15, 0.2)

        self.joint_angles = np.zeros(7)

        # goal location
        self.goal_position = (1.5, 2.5)

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
            math.cos(goal_theta - self.odometry.theta))) > 0.05:
            output_theta = self.pidTheta.update(self.odometry.theta, goal_theta, self.time.time())
            self.create.drive_direct(int(+output_theta), int(-output_theta))
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
                self.create.drive_direct(0, 0)
                break
            self.sleep(0.01)
        self.pf.move_by(self.odometry.x - old_x, self.odometry.y - old_y, self.odometry.theta - old_theta)

    def visualize(self):
        x, y, theta = self.pf.get_estimate()
        self.virtual_create.set_pose((x, y, 0.1), theta)
        data = []
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

    def run(self):
        self.create.start()
        self.create.safe()

        self.create.drive_direct(0, 0)
        # build the map
        self.create.sim_get_position()
        self.time.sleep(2)
        start_position = self.create.sim_get_position()
        #print(start_position)
        
        self.odometry.x = start_position[0]
        self.odometry.y = start_position[1]
        self.odometry.theta = 0
        start_position = self.real_to_image(start_position)
        self.rrt.build(start_position, 6000, 6)
        goal = self.rrt.nearest_neighbor(self.real_to_image(self.goal_position))
        path = self.rrt.shortest_path(goal)
        #print("Path length: {}".format(len(path)))
        base_speed = 100
        
        # this is for visualize path
        for v in self.rrt.T:
            for u in v.neighbors:
                self.map.draw_line((v.state[0], v.state[1]), (u.state[0], u.state[1]), (0,0,0))
        for idx in range(0, len(path)-1):
            self.map.draw_line((path[idx].state[0], path[idx].state[1]), (path[idx+1].state[0], path[idx+1].state[1]), (0,255,0))
        self.map.save("lab10_rrt.png")

        self.virtual_create.set_pose((1.5, 1.5, 0.1), 0)

        # request sensors
        self.create.start_stream([
            create2.Sensor.LeftEncoderCounts,
            create2.Sensor.RightEncoderCounts,
        ])
        self.visualize()
        self.virtual_create.enable_buttons()
        self.visualize()
        
        count = 0
        
        # execute the path
        for p in path:
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
                    if distance < 0.1:
                        self.create.drive_direct(0, 0)
                        break
                    self.pf.move_by(self.odometry.x - old_x, self.odometry.y - old_y, self.odometry.theta - old_theta)
            if count % 3 == 0:
                distance = self.sonar.get_distance()
                self.pf.measure(distance, 0)

                x, y, theta = self.pf.get_estimate()
                error_pos = self.distance((x, y), (self.odometry.x, self.odometry.y))
                error_theta = abs(theta - self.odometry.theta)
                
                if error_pos < 0.05:
                    self.odometry.x = x
                    self.odometry.y = y
                
                #print("Error: {}, {}, Estimate pos: ({}, {}, {}), odometry pos: ({}, {}, {})".format(error_pos, error_theta, x, y, theta, self.odometry.x, self.odometry.y, self.odometry.theta))
            self.visualize()
            count += 1

        reached_goal_location = (self.odometry.x, self.odometry.y)
        print(reached_goal_location)
        while True:
            shit = 1
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