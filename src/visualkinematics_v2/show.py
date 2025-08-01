from __future__ import annotations
import numpy as np
from ipywidgets import *
from pythreejs import *
import time
from scipy.spatial.transform import Rotation as R, Slerp
from numba import njit
from typing import *
import threading
import ipyevents
import visualkinematics_v2.manipulator
from visualkinematics_v2 import util
from IPython.display import display




class Inspector:
    def __init__(self, environment:Environment, manipulator:visualkinematics_v2.manipulator.Manipulator):
        self.view:InspectorView = InspectorView(env=environment, manipulator=manipulator)
        self.controller:Inspector_Controller = Inspector_Controller(view=self.view, environment=environment, manipulator=manipulator)





class Inspector_Controller:
    mouse_down:bool = False

    def __init__(self, view:InspectorView, environment, manipulator):
        self.view:InspectorView = view
        self.environment = environment
        self.manipulator:visualkinematics_v2.manipulator.Manipulator = manipulator
        self.trans_step = 0.05
        self.rot_step = 4.0
        self.button_wait_time = 0.01

        self.lower_limits = []
        self.upper_limits = []

        view.take_position_button.on_click(self._on_click_take_position_button)
        view.save_button.on_click(self._on_click_save_button)

        ipyevents.Event(source=view.trans_x_minus, watched_events=['mousedown', 'mouseup']).on_dom_event(self._on_trans_x_minus_button)
        ipyevents.Event(source=view.trans_x_plus, watched_events=['mousedown', 'mouseup']).on_dom_event(self._on_trans_x_plus_button)
        ipyevents.Event(source=view.trans_y_minus, watched_events=['mousedown', 'mouseup']).on_dom_event(self._on_trans_y_minus_button)
        ipyevents.Event(source=view.trans_y_plus, watched_events=['mousedown', 'mouseup']).on_dom_event(self._on_trans_y_plus_button)
        ipyevents.Event(source=view.trans_z_minus, watched_events=['mousedown', 'mouseup']).on_dom_event(self._on_trans_z_minus_button)
        ipyevents.Event(source=view.trans_z_plus, watched_events=['mousedown', 'mouseup']).on_dom_event(self._on_trans_z_plus_button)

        ipyevents.Event(source=view.rot_x_minus, watched_events=['mousedown', 'mouseup']).on_dom_event(self._on_rot_x_minus_button)
        ipyevents.Event(source=view.rot_x_plus, watched_events=['mousedown', 'mouseup']).on_dom_event(self._on_rot_x_plus_button)
        ipyevents.Event(source=view.rot_y_minus, watched_events=['mousedown', 'mouseup']).on_dom_event(self._on_rot_y_minus_button)
        ipyevents.Event(source=view.rot_y_plus, watched_events=['mousedown', 'mouseup']).on_dom_event(self._on_rot_y_plus_button)
        ipyevents.Event(source=view.rot_z_minus, watched_events=['mousedown', 'mouseup']).on_dom_event(self._on_rot_z_minus_button)
        ipyevents.Event(source=view.rot_z_plus, watched_events=['mousedown', 'mouseup']).on_dom_event(self._on_rot_z_plus_button)



        for joint, theta_rot_slider in view.joints_sliders.items():
            self.connect_sliders(joint=joint, theta_rot_slider=theta_rot_slider)
            self.lower_limits.append(joint.lower_limit)
            self.upper_limits.append(joint.upper_limit)




        r = R.from_quat(list(self.manipulator.tcp_target.quaternion))
        euler = r.as_euler('zyx', degrees=False)
        self.set_rotation_text(euler)
        self.set_position_text(self.manipulator.tcp_target.position)



    def connect_sliders(self, joint, theta_rot_slider):
        #display(f"joint:{joint.name}    slider:{theta_rot_slider.value}")
        renderable = joint
        if hasattr(joint, "get_renderable"):
            renderable = joint.get_renderable()

        rot = R.from_quat(list(renderable.quaternion))
        euler = rot.as_euler("XYZ", degrees=True)
        
        sign = 1
        if sum(joint.axis) < 0 : sign *= -1
        z = joint.dh_alignment[:3, -2]     #vorletzte spalte, erste 3 elemente (Spaltenvektor z-achse)
        if sum(z) < 0 : sign *= -1

        def _on_rot_slider(change):
            if abs(joint.axis[0]) == 1:
                util.set_rotation(renderable, [theta_rot_slider.value * sign, euler[1], euler[2]], "XYZ")
            elif abs(joint.axis[1]) == 1:
                util.set_rotation(renderable, [euler[0], theta_rot_slider.value * sign, euler[2]], "XYZ")
            elif abs(joint.axis[2]) == 1:
                util.set_rotation(renderable, [euler[0], euler[1], theta_rot_slider.value * sign], "XYZ")
            transform = self.manipulator.get_global_tcp_transform()
            util.apply_transformation_matrix(self.manipulator.tcp_target, transform)
            
            def worker():
                position_vector = transform[:3, 3]
                rotation_matrix = transform[:3, :3]
                r = R.from_matrix(rotation_matrix)
                euler = r.as_euler('zyx', degrees=False)
                self.set_rotation_text(euler)
                self.set_position_text(position_vector)
            
            thread = threading.Thread(target=worker)
            thread.start()



        def handle_event(event):
            if event['type'] == 'mouseenter':
                theta_rot_slider.observe(_on_rot_slider, names="value")
            elif event['type'] == 'mouseleave':
                theta_rot_slider.unobserve(_on_rot_slider, names="value")
                
        theta_rot_slider.observe(_on_rot_slider, names="value")
        theta_rot_slider.unobserve(_on_rot_slider, names="value")

        mouse_events = ipyevents.Event(
        source = theta_rot_slider,
        watched_events=['mouseenter', 'mouseleave'],
        prevent_default_action=True
        )
        mouse_events.on_dom_event(handle_event)


    def rot_x(self, change):
        if self.view.local_space_check_box.value:
            util.rotate(self.manipulator.tcp_target, [change, 0, 0], "XYZ")
        else:
            util.rotate_global(self.manipulator.tcp_target, [change, 0, 0], "XYZ")
        r = R.from_quat(list(self.manipulator.tcp_target.quaternion))
        euler = r.as_euler('zyx', degrees=False)
        self.set_rotation_text(euler)


    def rot_y(self, change):
        if self.view.local_space_check_box.value:
            util.rotate(self.manipulator.tcp_target, [change, 0, 0], "YXZ")
        else:
            util.rotate_global(self.manipulator.tcp_target, [0, change, 0], "XYZ")
        r = R.from_quat(list(self.manipulator.tcp_target.quaternion))
        euler = r.as_euler('zyx', degrees=False)
        self.set_rotation_text(euler)


    def rot_z(self, change):
        if self.view.local_space_check_box.value:
            util.rotate(self.manipulator.tcp_target, [change, 0, 0], "ZYX")
        else:
            util.rotate_global(self.manipulator.tcp_target, [0, 0, change], "XYZ")
        r = R.from_quat(list(self.manipulator.tcp_target.quaternion))
        euler = r.as_euler('zyx', degrees=False)
        self.set_rotation_text(euler)
    



    def trans(self, v):#noch fehler drin
        rot_mat = R.from_quat(list(self.manipulator.tcp_target.quaternion)).as_matrix()
        if self.view.local_space_check_box.value:
            final_v = rot_mat @ v
        else : final_v = v
        self.manipulator.tcp_target.position = tuple(np.array([self.manipulator.tcp_target.position[0], self.manipulator.tcp_target.position[1], self.manipulator.tcp_target.position[2]]) + np.array(final_v))
        self.set_position_text(self.manipulator.tcp_target.position)


    def set_position_text(self, pos_vec):
        self.view.x_current_pos_text.value = f"{pos_vec[0]}"
        self.view.y_current_pos_text.value = f"{pos_vec[1]}"
        self.view.z_current_pos_text.value = f"{pos_vec[2]}"
        

    def set_rotation_text(self, rot_vec):
        self.view.x_current_rot_text.value = f"{rot_vec[2]}"
        self.view.y_current_rot_text.value = f"{rot_vec[1]}"
        self.view.z_current_rot_text.value = f"{rot_vec[0]}"

    def _on_trans_x_minus_button(self, event):
        def down():
            while(self.__class__.mouse_down):
                self.trans([-self.trans_step,0,0])
                try:self.update_robot()
                except Exception as e: self.trans([self.trans_step,0,0])
                time.sleep(self.button_wait_time)

        if event['type'] == 'mousedown':
            self.__class__.mouse_down = True
            thread = threading.Thread(target=down)
            thread.start()
        elif event['type'] == 'mouseup':
            self.__class__.mouse_down = False

        

    def _on_trans_x_plus_button(self, event):
        def down():
            while(self.__class__.mouse_down):
                self.trans([self.trans_step,0,0])
                try:self.update_robot()
                except Exception as e: self.trans([-self.trans_step,0,0])
                time.sleep(self.button_wait_time)

        if event['type'] == 'mousedown':
            self.__class__.mouse_down = True
            thread = threading.Thread(target=down)
            thread.start()
        elif event['type'] == 'mouseup':
            self.__class__.mouse_down = False
        



    def _on_trans_y_minus_button(self, event):
        def down():
            while(self.__class__.mouse_down):
                self.trans([0,-self.trans_step,0])
                try:self.update_robot()
                except Exception as e: self.trans([0,self.trans_step,0])
                time.sleep(self.button_wait_time)

        if event['type'] == 'mousedown':
            self.__class__.mouse_down = True
            thread = threading.Thread(target=down)
            thread.start()
        elif event['type'] == 'mouseup':
            self.__class__.mouse_down = False




    def _on_trans_y_plus_button(self, event):
        def down():
            while(self.__class__.mouse_down):
                self.trans([0,self.trans_step,0])
                try:self.update_robot()
                except Exception as e: self.trans([0,-self.trans_step,0])
                time.sleep(self.button_wait_time)

        if event['type'] == 'mousedown':
            self.__class__.mouse_down = True
            thread = threading.Thread(target=down)
            thread.start()
        elif event['type'] == 'mouseup':
            self.__class__.mouse_down = False




    def _on_trans_z_minus_button(self, event):
        def down():
            while(self.__class__.mouse_down):
                self.trans([0,0,-self.trans_step])
                try:self.update_robot()
                except Exception as e: self.trans([0,0,self.trans_step])
                time.sleep(self.button_wait_time)

        if event['type'] == 'mousedown':
            self.__class__.mouse_down = True
            thread = threading.Thread(target=down)
            thread.start()
        elif event['type'] == 'mouseup':
            self.__class__.mouse_down = False




    def _on_trans_z_plus_button(self, event):
        def down():
            while(self.__class__.mouse_down):
                self.trans([0,0,self.trans_step])
                try:self.update_robot()
                except Exception as e: self.trans([0,0,-self.trans_step])
                time.sleep(self.button_wait_time)

        if event['type'] == 'mousedown':
            self.__class__.mouse_down = True
            thread = threading.Thread(target=down)
            thread.start()
        elif event['type'] == 'mouseup':
            self.__class__.mouse_down = False



    def _on_rot_x_minus_button(self, event):
        def down():
            while(self.__class__.mouse_down):
                self.rot_x(self.rot_step)
                try:self.update_robot()
                except Exception as e: self.rot_x(-self.rot_step)
                time.sleep(self.button_wait_time)

        if event['type'] == 'mousedown':
            self.__class__.mouse_down = True
            thread = threading.Thread(target=down)
            thread.start()
        elif event['type'] == 'mouseup':
            self.__class__.mouse_down = False


    def _on_rot_x_plus_button(self, event):
        def down():
            while(self.__class__.mouse_down):
                self.rot_x(self.rot_step)
                try:self.update_robot()
                except Exception as e: self.rot_x(-self.rot_step)
                time.sleep(self.button_wait_time)

        if event['type'] == 'mousedown':
            self.__class__.mouse_down = True
            thread = threading.Thread(target=down)
            thread.start()
        elif event['type'] == 'mouseup':
            self.__class__.mouse_down = False
        



    def _on_rot_y_minus_button(self, event):
        def down():
            while(self.__class__.mouse_down):
                self.rot_y(-self.rot_step)
                try:self.update_robot()
                except Exception as e: self.rot_y(self.rot_step)
                time.sleep(self.button_wait_time)

        if event['type'] == 'mousedown':
            self.__class__.mouse_down = True
            thread = threading.Thread(target=down)
            thread.start()
        elif event['type'] == 'mouseup':
            self.__class__.mouse_down = False




    def _on_rot_y_plus_button(self, event):
        def down():
            while(self.__class__.mouse_down):
                self.rot_y(self.rot_step)
                try:self.update_robot()
                except Exception as e: self.rot_y(-self.rot_step)
                time.sleep(self.button_wait_time)

        if event['type'] == 'mousedown':
            self.__class__.mouse_down = True
            thread = threading.Thread(target=down)
            thread.start()
        elif event['type'] == 'mouseup':
            self.__class__.mouse_down = False




    def _on_rot_z_minus_button(self, event):
        def down():
            while(self.__class__.mouse_down):
                self.rot_z(-self.rot_step)
                try:self.update_robot()
                except Exception as e: self.rot_z(self.rot_step)
                time.sleep(self.button_wait_time)

        if event['type'] == 'mousedown':
            self.__class__.mouse_down = True
            thread = threading.Thread(target=down)
            thread.start()
        elif event['type'] == 'mouseup':
            self.__class__.mouse_down = False




    def _on_rot_z_plus_button(self, event):
        def down():
            while(self.__class__.mouse_down):
                self.rot_z(self.rot_step)
                try:self.update_robot()
                except Exception as e: self.rot_z(-self.rot_step)
                time.sleep(self.button_wait_time)

        if event['type'] == 'mousedown':
            self.__class__.mouse_down = True
            thread = threading.Thread(target=down)
            thread.start()
        elif event['type'] == 'mouseup':
            self.__class__.mouse_down = False






    def update_robot(self):
        r = R.from_quat(list(self.manipulator.tcp_target.quaternion)).as_matrix()
        p = self.manipulator.tcp_target.position
        q0 = np.array(list(self.manipulator.dh.joint_angles.values()), float)
        try:
            q_sol = self.manipulator.dh.inverse_kinematics6D_with_limits(p, r, q0, 20, 0.0001, self.lower_limits, self.upper_limits)
            self.manipulator.animate_by_theta(q_sol, 0.04, 1, True, False)
            self.manipulator.update_dh_angles()
            for slider, angle in zip(list(self.view.joints_sliders.values()), list(self.manipulator.dh.joint_angles.values())):
                slider.value = angle / (2*np.pi) * 360.0
        except ValueError as e:
            self.environment.add_info(f"{e}")
            raise Exception
        #------
        #for (name, slider), q in zip(self.sliders.items(), q_sol):
            #slider.value = (q / (2*np.pi)) * 360.0


            
        
                




    def _on_click_take_position_button(self, button:widgets.Button):
        try:
            button.description = "Pose Einnehmen"
            button.icon='pause'
            self.manipulator.animate_by_learned_pose(name = self.pose_dropdown.value, synchronous=True, duation=4)
            
        except Exception as e:
            info = self.manipulator.environment.add_info(f"Beim Einnehmen der Pose ist ein Fehler aufgetreten!: {e}")
        button.description = "Pose Einnehmen"
        button.icon='play'







    def _on_click_save_button(self, button:widgets.Button):
        try:
            self.manipulator.learn(pose_name = self.save_pose_textfield.value)
            opts = list(self.pose_dropdown.options)
            opts.append(self.save_pose_textfield.value)
            self.pose_dropdown.options = opts
            button.description = "Gespeichert!"
            button.icon='check'
        
        except Exception as e:
            self.manipulator.environment.add_info(f"Beim Speichern der Pose ist ein Fehler aufgetreten!: {e}")
        def reset():
            time.sleep(1.5)
            button.description = "Pose Speichern"
            button.icon='save'
        threading.Thread(target=reset).start() 





class InspectorView:

    teach_section_layout = widgets.Layout(
        #border='1px solid gray',
        margin='0px,0px,0px,0px',
        padding='2px 2px, 2px 0px',
        height='70px',
        max_width='302px',
        width='302',
        overflow='hidden',  # Scrollen deaktivieren
        flex='none'
        )
    
    button_layout = widgets.Layout(width='30px', height='30px', margin='5px 5px 5px 5px')


    layout_box = widgets.Layout(
            #border='1px solid gray',
            padding='5px 5px 5px 5px',
            margin = '5px 5px 5px 5px',
            #width='100',
            max_height='800px',
            overflow='hidden',  # Scrollen deaktivieren
            flex='none'
        )
    


    layout_text = widgets.Layout(
            #border='1px solid gray',
            padding='5px 2px 5px 2px',
            margin = '5px 2px 5px 2px',
            max_width='65px',
            width = "65px",
            max_height='50px',
            overflow='hidden',  # Scrollen deaktivieren
            flex='none'
        )
    

    widget_layout = widgets.Layout(
            max_width='450px',
            overflow='hidden',  # Scrollen deaktivieren
            flex='none'
        )


    layout_gizmo_box = widgets.Layout(
            border='1px solid gray',
            margin = '5px 5px 5px 2px',
            padding='0px',
            max_width='300px',
            max_height='500px',
            overflow='hidden',  # Scrollen deaktivieren
            flex='none'
        )
    
    


    def __init__(self, env:Environment, manipulator:visualkinematics_v2.manipulator.Manipulator):
        #self.gizmo_controls:util.Environment.Gizmo_Controls = None
        self.environment = env
        self.sliders_signs = {}
        self.joints_sliders = {}
        self.manipulator = manipulator
        self.content = []
        self.pose_names = sorted(set(obj["name"] for obj in manipulator.learned_poses))
        self.pose_dropdown:widgets.Dropdown = widgets.Dropdown(
            options=self.pose_names,
            style={'description_width': 'initial'})

        self.save_pose_textfield:widgets.Text = widgets.Text(
            value='',
            placeholder='Name für Pose',
            #description='Eingabe:',
            disabled=False
        )
        self.save_button:widgets.Button = widgets.Button(
            description='Pose Speichern',
            disabled=False,
            button_style='',
            tooltip='Speichert Die Pose, sodass sie in Zukunft wieder eingenommen werden kann.',
            icon='save'
        )

        

        self.take_position_button:widgets.Button = widgets.Button(
            description='Pose Einnehmen',
            disabled=False,
            button_style='',
            tooltip='Überführt den Roboter in die Ausgewählte Pose!',
            icon='play'
        )

    

        hbox2 = widgets.VBox([self.pose_dropdown, self.take_position_button], layout=self.__class__.teach_section_layout)

        # Horizontal anordnen
        hbox1 = widgets.VBox([self.save_pose_textfield, self.save_button], layout=self.__class__.teach_section_layout)
        self.content.append(hbox1)
        self.content.append(hbox2)
        
        self._create_theta_sliders()

        if manipulator.dh is not None:
            util.apply_transformation_matrix(manipulator.tcp_target, manipulator.get_global_tcp_transform())
        env.add(manipulator.tcp_target)
        
        #-----------------------------------
        self.trans_x_minus = widgets.Button(description='-', layout=self.__class__.button_layout, disabled=False)
        trans_x_label = widgets.Label("X")
        self.trans_x_plus = widgets.Button(description='+', layout=self.__class__.button_layout, disabled=False)
        x_trans_box = widgets.HBox([self.trans_x_minus, trans_x_label, self.trans_x_plus])


        self.trans_y_minus = widgets.Button(description='-', layout=self.__class__.button_layout, disabled=False)
        trans_y_label = widgets.Label("Y")
        self.trans_y_plus = widgets.Button(description='+', layout=self.__class__.button_layout, disabled=False)
        y_trans_box = widgets.HBox([self.trans_y_minus, trans_y_label, self.trans_y_plus])


        self.trans_z_minus = widgets.Button(description='-', layout=self.__class__.button_layout, disabled=False)
        trans_z_label = widgets.Label("Z")
        self.trans_z_plus = widgets.Button(description='+', layout=self.__class__.button_layout, disabled=False)
        z_trans_box = widgets.HBox([self.trans_z_minus, trans_z_label, self.trans_z_plus])

        trans_box = widgets.VBox([x_trans_box, y_trans_box, z_trans_box], layout=self.__class__.layout_box)



        self.rot_x_minus = widgets.Button(description='-', layout=self.__class__.button_layout, disabled=False)
        rot_x_label = widgets.Label("R")
        self.rot_x_plus = widgets.Button(description='+', layout=self.__class__.button_layout, disabled=False)
        x_rot_box = widgets.HBox([self.rot_x_minus, rot_x_label, self.rot_x_plus])


        self.rot_y_minus = widgets.Button(description='-', layout=self.__class__.button_layout, disabled=False)
        rot_y_label = widgets.Label("N")
        self.rot_y_plus = widgets.Button(description='+', layout=self.__class__.button_layout, disabled=False)
        y_rot_box = widgets.HBox([self.rot_y_minus, rot_y_label, self.rot_y_plus])


        self.rot_z_minus = widgets.Button(description='-', layout=self.__class__.button_layout, disabled=False)
        rot_z_label = widgets.Label("G")
        self.rot_z_plus = widgets.Button(description='+', layout=self.__class__.button_layout, disabled=False)
        z_rot_box = widgets.HBox([self.rot_z_minus, rot_z_label, self.rot_z_plus])

        rot_box = widgets.VBox([x_rot_box, y_rot_box, z_rot_box], layout=self.__class__.layout_box)


        
        tcp_label = widgets.Label("TCP Pose")
        gizmo_box = widgets.HBox([trans_box, rot_box], layout=self.__class__.layout_gizmo_box)
        
        self.local_space_check_box = widgets.Checkbox(description="local space", value=False, disabled=False)
        self.content.append(tcp_label)
        self.content.append(gizmo_box)
        self.content.append(self.local_space_check_box)
        

        x_current_pos = widgets.Label("X:")
        self.x_current_pos_text = widgets.Text(value="-", layout=self.__class__.layout_text)

        y_current_pos = widgets.Label("Y:")
        self.y_current_pos_text = widgets.Text(value="-", layout=self.__class__.layout_text)

        z_current_pos = widgets.Label("Z:")
        self.z_current_pos_text = widgets.Text(value="-", layout=self.__class__.layout_text)

        x_current_rot = widgets.Label("R:")
        self.x_current_rot_text = widgets.Text(value="-", layout=self.__class__.layout_text)

        y_current_rot = widgets.Label("N:")
        self.y_current_rot_text = widgets.Text(value="-", layout=self.__class__.layout_text)

        z_current_rot = widgets.Label("G:")
        self.z_current_rot_text = widgets.Text(value="-", layout=self.__class__.layout_text)

        meter_label = widgets.Label("(m)")
        radiant_label = widgets.Label("(rad)")
        current_pos_box = widgets.HBox([x_current_pos, self.x_current_pos_text, y_current_pos, self.y_current_pos_text, z_current_pos, self.z_current_pos_text, meter_label])
        current_rot_box = widgets.HBox([x_current_rot, self.x_current_rot_text, y_current_rot, self.y_current_rot_text, z_current_rot, self.z_current_rot_text, radiant_label])
        box = VBox([current_pos_box, current_rot_box], layout=self.__class__.layout_gizmo_box)
        self.content.append(box)
        
        
        
        #---------------------------------

        #self.gizmo_controls = env.Gizmo_Controls(manipulator.tcp_target, True, True, False, "TCP-Target", 3, 3, 3, -3, -3, -3, widgets_vertical=True, continuous_update=True, callback=self._on_gizmo_controls)
        #self.content.append(self.gizmo_controls.widget)
        self.content.append(visualkinematics_v2.manipulator.fps_text)
        self.widget = widgets.VBox(children = self.content, layout=self.__class__.widget_layout)








    def _create_theta_sliders(self):
        num = 1
        for j in self.manipulator.joints:
            if j.is_mimicer: continue
            joint_and_slider_and_sign = self._create_theta_slider(num, value=0, joint=j)
            if joint_and_slider_and_sign != None: 
                self.content.append(joint_and_slider_and_sign[1])
                self.joints_sliders[joint_and_slider_and_sign[0]] = joint_and_slider_and_sign[1]
                self.sliders_signs[joint_and_slider_and_sign[0]] = joint_and_slider_and_sign[2]
            num += 1


    
    def _create_theta_slider(self, num, value, joint):
        if joint.axis is None : return None
        sign = 1
        if sum(joint.axis) < 0 : sign *= -1
        z = joint.dh_alignment[:3, -2]     #vorletzte spalte, erste 3 elemente (Spaltenvektor z-achse)
        if sum(z) < 0 : sign *= -1

        
        renderable = joint
        if hasattr(joint, "get_renderable"):
            renderable = joint.get_renderable()

        layout1 = widgets.Layout(
                border='1px solid gray',
                padding='2px',
                height='40px',
                overflow='hidden',  # Scrollen deaktivieren
                flex='none'
            )  
        min = -180
        max = 180
        if joint.lower_limit is not None:
            min = np.rad2deg(joint.lower_limit) * sign
        if joint.upper_limit is not None:
            max = np.rad2deg(joint.upper_limit) * sign
        if min > max:
            tmp = max
            max = min
            min = tmp

        theta_rot_slider = FloatSlider(min=min, max=max, step=0.1, description=f'Theta {num}', layout=layout1)
        rot = R.from_quat(list(renderable.quaternion))
        euler = rot.as_euler("XYZ", degrees=True) 
        
        if abs(joint.axis[0]) == 1: theta_rot_slider.value = euler[0]
        elif abs(joint.axis[1]) == 1: theta_rot_slider.value = euler[1]
        elif abs(joint.axis[2]) == 1: theta_rot_slider.value = euler[2]
        

        return [joint, theta_rot_slider, sign]

#--------------------------------------------------------------------------------------------------------

















class Environment:
    '''
    Eine 3D-Umgebung, die eine Szene mit Kamera, Lichtquellen, Achsen, Gittern und Widgets für die Interaktivität erstellt.

    Diese Klasse erstellt eine 3D-Umgebung mit einer Vielzahl von Features, darunter eine Kamera, Lichtquellen, Achsen- und Gitterdarstellung sowie Steuerungen zur Anpassung von Objekten in der Szene (z.B. Rotation, Skalierung, Translation).

    :param width: Die Breite der Ansicht in Pixeln (Standard: 700).
    :param height: Die Höhe der Ansicht in Pixeln (Standard: 500).
    :param frame: Ein 3D-Achsenobjekt, das als Referenzrahmen in der Szene hinzugefügt wird (Standard: create_axes(8, name="B")).
    :param grid: Ein Gitterobjekt, das in der Szene angezeigt wird (Standard: create_grid_XY(14, 0.5)).
    :param up: Die Richtung der "Oben"-Achse, die die Orientierung der Kamera bestimmt (Standard: [0, 0, 1]).

    Diese Klasse enthält Methoden, um:
    - die Sichtbarkeit von Gitter und Achsen zu steuern,
    - interaktive Widgets für Objekte zu erstellen (Translation, Rotation, Skalierung),
    - Objekte der Szene hinzuzufügen,
    - globale oder lokale Transformationen auf Objekte anzuwenden.

    Weitere Features:
    - Die Umgebung kann mit einer interaktiven Steuerung für Kamera und Objekte angezeigt werden.
    - Widgets für die Manipulation von Objekten (Translation, Rotation, Skalierung) können zur Szene hinzugefügt werden.
    '''

    info_container_layout = widgets.Layout(
                #border='1px solid gray',
                #padding='5px 2px 5px 2px',
                #margin = '5px 2px 5px 2px',
                max_height = '150px',
                #overflow='auto',  
                flex='none'
            )
    
    info_layout = widgets.Layout(
                max_height = '30px',
                height = '30px',
                flex='none',
                overflow='hidden',
            )

    def __init__(self, width=700, height=500, frame=None, grid=None, up=[0,0,1], widgets_on_bottom=False):
        '''
        Initialisiert eine neue 3D-Umgebung.

        :param width: Die Breite der Ansicht.
        :param height: Die Höhe der Ansicht.
        :param frame: Ein 3D-Achsenobjekt, das in der Szene hinzugefügt wird.
        :param grid: Ein Gitterobjekt, das in der Szene angezeigt wird.
        :param up: Die Richtung der "Oben"-Achse für die Kamera.
        '''
        if frame is None:
            frame = util.create_axes(8, name="B")
        if grid is None:
            grid = util.create_grid_XY(14,0.5)
        self.frame = frame
        self.grid = grid
        self.scene = Scene()
        self.scene.background = "#DDDDDD"
        self.camera:PerspectiveCamera = PerspectiveCamera(position=[8, 8, 8],aspect=width/height, fov=50)
        self.camera.up = up
        self.frame = frame
        self.grid = grid
        self.light = PointLight(color='white', intensity=1.5, position=[5, 5, 5])
        self.scene.add([self.camera, self.light, self.frame, self.grid, AmbientLight(intensity=0.5)])
        self.children = []
        self.widgets_on_bottom = widgets_on_bottom
        # Renderer mit Orbit-Steuerung
        self.renderer = Renderer(camera=self.camera, scene=self.scene, controls=[OrbitControls(controlling=self.camera)], width=width, height=height, background_color="#87CEEB", background_opacity=1.0, antialias=True, precision='highp')
        self.frame_widgets = True
        self.widgets = []
        self.gizmo_controls:list[Gizmo_Controls] = []
        self.inspectors:list[Inspector] = []

        self.info_container:widgets.VBox = widgets.VBox(layout=self.__class__.info_container_layout)
        


        
        


    def toggle_grid(self, change):
        '''
        Schaltet die Sichtbarkeit des Gitters um.

        :param change: Das Ereignis, das diese Funktion auslöst (wird nicht genutzt).
        '''
        self.grid.visible = not self.grid.visible




    def toggle_axes(self, change):
        '''
        Schaltet die Sichtbarkeit der Achsen um.

        :param change: Das Ereignis, das diese Funktion auslöst (wird nicht genutzt).
        '''
        self.frame.visible = not self.frame.visible




    def _ipython_display_(self):
        '''
        Zeigt die Umgebung mit Renderer und interaktiven Widgets an, wenn sie in einem Jupyter-Notebook verwendet wird.
        '''
        mainbox_layout = widgets.Layout(
            #border='1px solid gray',
            padding='5px',
            flex='1 1 auto',     # <- erlaubt Schrumpfen & Wachsen
            min_width='0px',     # <- wichtig für Schrumpfen!
            overflow='auto',
            justify_content='center',
            align_items='center'
        )

        renderer_box_layout = widgets.Layout(
            #border='1px solid gray',
            #max_width='70vw',
            #max_height='70vh',
            overflow='auto',
            flex='1 1 auto',
            
        )

        b_layout = widgets.Layout(
            #border='1px solid gray',
            overflow='hidden',
            width='100%',
            #align_items='center',  #ist falsch!
        )


        outer_layout = widgets.Layout(
            #flex="1 1 auto",
            min_width='300px'
        )

        frame_widget_box_layout = widgets.Layout(
            width=f'{self.renderer.width}px'
        )


        renderer_box = HBox([self.renderer], layout=renderer_box_layout)

        mainbox = VBox([renderer_box], layout = mainbox_layout)
        if self.frame_widgets:
            checkbox_grid = Checkbox(value=True, description='Show Grid')
            checkbox_axes = Checkbox(value=True, description='Show Axes')
            #interactive_control_scale = widgets.interactive(update_cube_scale, x=self.x_scale_slider, y=self.y_scale_slider, z=self.z_scale_slider)
            checkbox_grid.observe(self.toggle_grid, names='value')
            checkbox_axes.observe(self.toggle_axes, names='value')
            frame_widget_box = HBox([checkbox_grid, checkbox_axes], layout=frame_widget_box_layout)
            mainbox.children = mainbox.children + (frame_widget_box,)


        widget_box_layout = widgets.Layout(
            #border='1px solid gray',
            padding='5px',
            height=f'{self.renderer.height}px',
            min_width='335px',
            max_width='1235px',
            align_items='center',
            overflow_y='auto',
            #overflow_x = "auto",
            flex="0 0 auto"
        )

        

        widget_box = VBox(children=[], layout = widget_box_layout)
        for w in self.widgets:
            widget_box.children = widget_box.children + (w,)
            w.layout.overvlow="hidden"

        #lol_box = HBox(children = [widget_box], layout=layout2)
        if self.widgets_on_bottom:
            b = VBox(children = [mainbox, widget_box], layout=b_layout)
        else:
            b = HBox(children = [mainbox, widget_box], layout=b_layout)
        outer_box = VBox([b, self.info_container], layout=outer_layout)
        display(outer_box)


        
    def set_frame_widgets(self, bool):
        '''
        Aktiviert oder deaktiviert die Anzeige von Frame-Widgets.

        :param bool: Wenn True, werden die Widgets angezeigt, andernfalls ausgeblendet.
        '''
        self.frame_widgets = bool


    def add(self, objects):
        """
        Fügt ein oder mehrere Objekte zur Environment hinzu.

        Die Objekte müssen entweder:
        - ein Attribut oder eine Methode `get_renderable()` besitzen, das ein pythreejs-kompatibles Objekt zurückgibt,
        - oder selbst direkt ein pythreejs-kompatibles Objekt sein (z. B. Mesh, Group, Line, etc.).

        Hinweis: Es erfolgt keine explizite Typprüfung. Es wird davon ausgegangen, dass übergebene Objekte entweder
        direkt von pythreejs unterstützt werden oder über `get_renderable()` ein entsprechendes Objekt liefern.

        :param objekts: Ein einzelnes Objekt oder eine Liste von Objekten, die zur Environment hinzugefügt werden sollen.
        :raises AttributeError: Wenn `get_renderable()` aufgerufen wird, aber nicht vorhanden ist.
        :raises TraitError / TypeError: Wenn ein nicht unterstützter Objekttyp der Szene hinzugefügt wird.
        """
        if hasattr(objects, "set_environment") : objects.set_environment(self)
        if isinstance(objects, Object3D):
            self.scene.add(objects)
            self.children.append(objects)
        elif hasattr(objects, "get_renderable"):
            self.scene.add(objects.get_renderable())
            self.children.append(objects)
        elif isinstance(objects, Iterable):
            for obj in objects:
                if hasattr(obj, "set_environment") : obj.set_environment(self)
                if isinstance(obj, Object3D):
                    self.scene.add(obj)
                    self.children.append(obj)
                elif hasattr(obj, "get_renderable"):
                    self.scene.add(obj.get_renderable())
                    self.children.append(obj)




    def add_widget(self, widget):
        '''
        Fügt ein Widget zur Umgebung hinzu. Dabei kann es sich auch um ein Buendel von Widgets in einer HBox oder einer VBox handeln.

        :param widget: Das Widget, das der Umgebung hinzugefügt werden soll.
        '''
        self.widgets.append(widget)
    
    def add_info(self, info_text:str):
        test_button:widgets.Button = widgets.Button(
            description='',
            tooltip='',
            icon='times',
            layout=widgets.Layout(width='32px')
        )
        
        test_label = widgets.HTML(value=f'<span style="font-size:14px; color:red;">{info_text}</span>')
        info = HBox(children=[test_label, test_button], layout=self.__class__.info_layout)
        self.info_container.children = list(self.info_container.children) + [info]
        def on_click(button):
            self.remove_info(info)
        test_button.on_click(on_click)
        return info

    def remove_info(self, info):
        self.info_container.children = tuple(w for w in self.info_container.children if w != info)
    


    def add_gizmo_controls(
            self, obj, translation=True, rotation=True, scale=False, name="",
            max_trans_x:float = 1, max_trans_y:float = 1, max_trans_z:float = 1,
            min_trans_x:float = -1, min_trans_y:float = -1, min_trans_z:float = -1,
            widgets_vertical:Bool = False,
            continuous_update=True,
            callback: Callable[[], None] = None):
        '''
        Fügt ein Gizmo-Steuerelement zur Manipulation eines Objekts in der Umgebung hinzu (Translation, Rotation, Skalierung).

        :param obj: Das Objekt, das manipuliert werden soll.
        :param translation: Wenn True, werden Schieberegler für die Translation angezeigt.
        :param rotation: Wenn True, werden Schieberegler für die Rotation angezeigt.
        :param scale: Wenn True, werden Schieberegler für die Skalierung angezeigt.
        '''
        controls = Gizmo_Controls(obj, translation, rotation, scale, name,
                                    max_trans_x, max_trans_y, max_trans_z,
                                    min_trans_x, min_trans_y, min_trans_z,
                                    widgets_vertical, continuous_update, callback)
        self.gizmo_controls.append(controls)
        self.add_widget(controls.view.widget)
        return controls


    def add_inspector(self, obj):
        inspector = Inspector(self, obj)
        self.inspectors.append(inspector)
        self.add_widget(inspector.view.widget)
        return inspector




#----------------------------------------GIZMO_CONTROLS---------------------------------------------------------

class Gizmo_Controls_Controller:

    def __init__(self, view, obj, translation=True, rotation=True, scale=False, name="",
            max_trans_x:float = 1, max_trans_y:float = 1, max_trans_z:float = 1,
            min_trans_x:float = -1, min_trans_y:float = -1, min_trans_z:float = -1,
            widgets_vertical:Bool = False,
            continuous_update=True,
            callback: Callable[[], None] = None):
        self.view:Gizmo_Controls_View = view
        self.callback = callback
        self.obj = obj

        self.obj_renderable = obj
        if hasattr(obj, "get_renderable"):
            self.obj_renderable = obj.get_renderable()

        if translation:
            self.view.x_trans_slider.observe(self._on_trans_slider, names='value')
            self.view.y_trans_slider.observe(self._on_trans_slider, names="value")
            self.view.z_trans_slider.observe(self._on_trans_slider, names="value")
        if rotation:
            self.view.x_rot_slider.observe(self._on_rot_slider, names="value")
            self.view.y_rot_slider.observe(self._on_rot_slider, names="value")
            self.view.z_rot_slider.observe(self._on_rot_slider, names="value")
        if scale:
            self.view.x_scale_slider.observe(self._on_scale_slider, names="value")
            self.view.y_scale_slider.observe(self._on_scale_slider, names="value")
            self.view.z_scale_slider.observe(self._on_scale_slider, names="value")

        self.view.rotation_order_dropdown.observe(self._on_rotation_order_change, names='value')  
        self.view.local_space_check_box.observe(self._on_local_space_check_box, names='value')





    def _on_scale_slider(self, change):
        util.set_scale(self.obj_renderable, [self.view.x_scale_slider.value, self.view.y_scale_slider.value, self.view.z_scale_slider.value])
        if self.callback is not None : self.callback()



    def _on_trans_slider(self, change):#noch fehler drin
        delta = change["new"] - change["old"]
        v = None
        if change['owner'] is self.view.x_trans_slider: v = np.array([delta, 0, 0])
        if change['owner'] is self.view.y_trans_slider: v = np.array([0, delta, 0])
        if change['owner'] is self.view.z_trans_slider: v = np.array([0, 0, delta])
        rot_mat = R.from_quat(list(self.obj_renderable.quaternion)).as_matrix()
        if self.view.local_space_check_box.value:
            final_v = rot_mat @ v
        else : final_v = v
        self.obj_renderable.position = tuple(np.array([self.obj_renderable.position[0], self.obj_renderable.position[1], self.obj_renderable.position[2]]) + np.array(final_v))
        if self.callback is not None : self.callback()


    

    def _on_rot_slider(self, change):
        o = self.view.rotation_order_dropdown.value
        if (o == "zyz" or o == "ZYZ" or
            o == "xyx" or o == "XYX" or
            o == "xzx" or o == "XZX" or
            o == "yxy" or o == "YXY" or
            o == "yzy" or o == "YZY" or
            o == "zxz" or o == "ZXZ"):
            if self.view.local_space_check_box.value:
                util.set_rotation(self.obj_renderable, [self.view.x_rot_slider.value, self.view.y_rot_slider.value, self.view.z_rot_slider.value], self.view.rotation_order_dropdown.value)
            else:
                util.set_rotation_global(self.obj_renderable, [self.view.x_rot_slider.value, self.view.y_rot_slider.value, self.view.z_rot_slider.value], self.view.rotation_order_dropdown.value)
        else:
            angles = util.order_angles([self.view.x_rot_slider.value, self.view.y_rot_slider.value, self.view.z_rot_slider.value], "XYZ", self.view.rotation_order_dropdown.value)
            if self.view.local_space_check_box.value:
                util.set_rotation(self.obj_renderable, angles, self.view.rotation_order_dropdown.value)
            else:
                util.set_rotation_global(self.obj_renderable, angles, self.view.rotation_order_dropdown.value)
        if self.callback is not None : self.callback()



    def _on_local_space_check_box(self, change):
        self.view.x_rot_slider.unobserve(self._on_rot_slider, names="value")
        self.view.y_rot_slider.unobserve(self._on_rot_slider, names="value")
        self.view.z_rot_slider.unobserve(self._on_rot_slider, names="value")
        if self.view.local_space_check_box.value:
            euler = util.quaternion_to_euler(self.obj.quaternion[0], self.obj.quaternion[1], self.obj.quaternion[2], self.obj.quaternion[3], self.view.rotation_order_dropdown.value)
            euler = util.order_angles(euler, self.view.rotation_order_dropdown.value, "XYZ")
            self.view.x_rot_slider.value = euler[0]
            self.view.y_rot_slider.value = euler[1]
            self.view.z_rot_slider.value = euler[2]
        else:
            euler = util.quaternion_to_euler(self.obj.quaternion[0], self.obj.quaternion[1], self.obj.quaternion[2], self.obj.quaternion[3], self.view.rotation_order_dropdown.value[::-1])
            euler = util.order_angles(euler, self.view.rotation_order_dropdown.value[::-1], "XYZ")
            self.view.x_rot_slider.value = euler[0]
            self.view.y_rot_slider.value = euler[1]
            self.view.z_rot_slider.value = euler[2]
        self.view.x_rot_slider.observe(self._on_rot_slider, names="value")
        self.view.y_rot_slider.observe(self._on_rot_slider, names="value")
        self.view.z_rot_slider.observe(self._on_rot_slider, names="value")






    def _on_rotation_order_change(self, change):
        o = self.view.rotation_order_dropdown.value
        if (o=="ZYZ" or o=="zyz"):
            self.view.x_rot_slider.description="Rotate Z"
            self.view.y_rot_slider.description="Rotate Y"
            self.view.z_rot_slider.description="Rotate Z"
        elif (o=="XYX" or o=="xyx"):
            self.view.x_rot_slider.description="Rotate X"
            self.view.y_rot_slider.description="Rotate Y"
            self.view.z_rot_slider.description="Rotate X"
        elif (o=="XZX" or o=="xzx"):
            self.view.x_rot_slider.description="Rotate X"
            self.view.y_rot_slider.description="Rotate Z"
            self.view.z_rot_slider.description="Rotate X"
        elif (o=="YXY" or o=="yxy"):
            self.view.x_rot_slider.description="Rotate Y"
            self.view.y_rot_slider.description="Rotate X"
            self.view.z_rot_slider.description="Rotate Y"
        elif (o=="YZY" or o=="yzy"):
            self.view.x_rot_slider.description="Rotate Y"
            self.view.y_rot_slider.description="Rotate Z"
            self.view.z_rot_slider.description="Rotate Y"
        elif (o=="ZXZ" or o=="zxz"):
            self.view.x_rot_slider.description="Rotate Z"
            self.view.y_rot_slider.description="Rotate X"
            self.view.z_rot_slider.description="Rotate Z"
        else:
            self.view.x_rot_slider.description="Rotate X"
            self.view.y_rot_slider.description="Rotate Y"
            self.view.z_rot_slider.description="Rotate Z"
        self._on_rot_slider(None)




class Gizmo_Controls:
    def __init__(
            self, obj, translation=True, rotation=True, scale=False, name="",
            max_trans_x:float = 1, max_trans_y:float = 1, max_trans_z:float = 1,
            min_trans_x:float = -1, min_trans_y:float = -1, min_trans_z:float = -1,
            widgets_vertical:Bool = False,
            continuous_update=True,
            callback: Callable[[], None] = None):
        self.view:Gizmo_Controls_View = Gizmo_Controls_View(obj=obj, translation=translation, rotation=rotation, scale=scale, name=name,
                                                max_trans_x=max_trans_x, max_trans_y=max_trans_y, max_trans_z=max_trans_z,
                                                min_trans_x=min_trans_x, min_trans_y=min_trans_y, min_trans_z=min_trans_z,
                                                widgets_vertical=widgets_vertical, continuous_update=continuous_update, callback=callback)
        self.controller:Gizmo_Controls_Controller = Gizmo_Controls_Controller(view=self.view, obj=obj, translation=translation, rotation=rotation, scale=scale, name=name,
                                                max_trans_x=max_trans_x, max_trans_y=max_trans_y, max_trans_z=max_trans_z,
                                                min_trans_x=min_trans_x, min_trans_y=min_trans_y, min_trans_z=min_trans_z,
                                                widgets_vertical=widgets_vertical, continuous_update=continuous_update, callback=callback)   




class Gizmo_Controls_View:
    layout1_horizontal = widgets.Layout(
            #border='1px solid gray',
            padding='5px',
            #width='100',
            height='125px',
            overflow='hidden',  # Scrollen deaktivieren
            flex='none'
            )
        
    layout1_vertical = widgets.Layout(
            #border='1px solid gray',
            padding='5px',
            #width='100',
            #height='125px',
            overflow='hidden',  # Scrollen deaktivieren
            flex='none'
        )
    
    layout2_horizontal = widgets.Layout(
            border='1px solid gray',
            padding='5px',
            #width='100',
            height='200px',
            overflow='hidden',  # Scrollen deaktivieren
            flex='none'
        )
    
    layout2_vertical = widgets.Layout(
            border='1px solid gray',
            padding='5px',
            #width='100',
            max_height='500px',
            overflow='hidden',  # Scrollen deaktivieren
            flex='none'
        )


    def __init__(
            self, obj, translation=True, rotation=True, scale=False, name="",
            max_trans_x:float = 1, max_trans_y:float = 1, max_trans_z:float = 1,
            min_trans_x:float = -1, min_trans_y:float = -1, min_trans_z:float = -1,
            widgets_vertical:Bool = False,
            continuous_update=True,
            callback: Callable[[], None] = None):
        '''
        Erstellt ein Gizmo-Steuerelement zur Manipulation eines Objekts (Translation, Rotation, Skalierung).

        :param obj: Das Objekt, das manipuliert werden soll.
        :param translation: Wenn True, werden Schieberegler für die Translation angezeigt.
        :param rotation: Wenn True, werden Schieberegler für die Rotation angezeigt.
        :param scale: Wenn True, werden Schieberegler für die Skalierung angezeigt.
        '''
        
        self.callback = callback
        self.obj = obj
        self.content = []
        
        if name != "":
            self.content.append(Label(name))
        self.obj_renderable = obj
        if hasattr(obj, "get_renderable"):
            self.obj_renderable = obj.get_renderable()

        
        if widgets_vertical:
            layout1 = self.__class__.layout1_vertical
            layout2 = self.__class__.layout2_vertical
        else :    
            layout1 = self.__class__.layout1_horizontal
            layout2 = self.__class__.layout2_horizontal

        if translation:
            self.x_trans_slider = FloatSlider(min=min_trans_x, max = max_trans_x, step=0.001, description="Translation X", continuous_update=continuous_update)
            self.y_trans_slider = FloatSlider(min=min_trans_y, max = max_trans_y, step=0.001, description="Translation Y", continuous_update=continuous_update)
            self.z_trans_slider = FloatSlider(min=min_trans_z, max = max_trans_z, step=0.001, description="Translation Z", continuous_update=continuous_update)
            self.x_trans_slider.value = self.obj_renderable.position[0]
            self.y_trans_slider.value = self.obj_renderable.position[1]
            self.z_trans_slider.value = self.obj_renderable.position[2]

            trans_box = VBox(children=[self.x_trans_slider, self.y_trans_slider, self.z_trans_slider], layout=layout1)
            
            self.content.append(trans_box)


        if rotation:
            self.x_rot_slider = FloatSlider(min=-180, max=180, step=0.1, description='Rotate X', continuous_update=continuous_update)
            self.y_rot_slider = FloatSlider(min=-180, max=180, step=0.1, description='Rotate Y', continuous_update=continuous_update)
            self.z_rot_slider = FloatSlider(min=-180, max=180, step=0.1, description='Rotate Z', continuous_update=continuous_update)
            rot = R.from_quat(list(self.obj_renderable.quaternion))
            euler = rot.as_euler("XYZ", degrees=True) 
            self.x_rot_slider.value = euler[0]
            self.y_rot_slider.value = euler[1]
            self.z_rot_slider.value = euler[2]
            rot_box = VBox(children=[self.x_rot_slider, self.y_rot_slider, self.z_rot_slider], layout=layout1)
            self.content.append(rot_box)


            

        if scale:
            self.x_scale_slider = FloatSlider(min=0, max=5, step=0.001, description="Scale X", value=1, continuous_update=continuous_update)
            self.y_scale_slider = FloatSlider(min=0, max=5, step=0.001, description="Scale Y", value=1, continuous_update=continuous_update)
            self.z_scale_slider = FloatSlider(min=0, max=5, step=0.001, description="Scale Z", value=1, continuous_update=continuous_update)
            scale_box = VBox(children=[self.x_scale_slider, self.y_scale_slider, self.z_scale_slider], layout=layout1)
            self.content.append(scale_box)





        if rotation:
            #ZYX ist Roll Nick Gier wie in der Vorlesung, ZYZ ist Euler wie in der Vorlesung
            self.rotation_order_dropdown = Dropdown(
                options=['XYZ', 'XZY', 'YXZ', 'YZX', 'ZXY', 'ZYX', "ZYZ", "XYX", "XZX", "YXY", "YZY", "ZXZ"],
                value='XYZ',
                description='Rotation Order:',
            )

            
            


        self.local_space_check_box = widgets.Checkbox(value=False, description="Lokale Transformation", layout=widgets.Layout(width='350px', height='30px'))

        if widgets_vertical:
            box = VBox(children = self.content, layout = layout1)
        else:
            box = HBox(children = self.content, layout = layout1)
        if rotation :
            main_box = VBox(children = [box, self.rotation_order_dropdown, self.local_space_check_box], layout=layout2)
        else :
            main_box = VBox(children = [box, self.local_space_check_box], layout=layout2)
        self.widget = main_box
            
        


    