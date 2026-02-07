import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from rm_rl.assets import ISAACLAB_ASSETS_DATA_DIR

Infantry_25_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Infantry_25/usd/Infantry_25_headless.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=1
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.276),
        joint_pos={
            "hip.*": 1.05,
            "knee.*": 1.2845,                  
            "wheel.*": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "joint": DelayedPDActuatorCfg( 
            joint_names_expr=["hip.*", "knee.*"],
            effort_limit=80,
            velocity_limit=20,
            stiffness=40.0,
            damping=2.0,
            friction=0.0,
            # armature=0.016,
            armature=0.0,
            min_delay=0,
            max_delay=5,
        ),
        "wheel": DelayedPDActuatorCfg(
            joint_names_expr=["wheel.*"],
            effort_limit=5,
            velocity_limit=60,
            stiffness=0.0, 
            damping=0.2,
            friction=0.0,
            armature=0.002,
            min_delay=0,
            max_delay=5,
        ),
    },
)

