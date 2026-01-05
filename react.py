"""ReACT-style agent implementation with explicit Thought/Action/Observation loop

This implementation follows the ReACT (Reasoning and Acting) pattern:
1. While Loop: Continues until task is complete or early stop conditions are met
2. Thought/Action/Observation in one iteration: Each loop iteration includes:
   - Thought: Reasoning step extracted from LLM response
   - Action: Action extracted and executed
   - Observation: Result from environment after action execution
3. Uses all the same prompts as the original version via prompt_constructor

The key difference from the original run.py is that we explicitly track and log
the Thought/Action/Observation cycle for better interpretability.
"""
import argparse
import json
import logging
import os
import random
import subprocess
import tempfile
import time
from pathlib import Path

from agent import construct_agent, PromptAgent
from agent.prompts import *
from browser_env import (
    Action,
    ActionTypes,
    ScriptBrowserEnv,
    StateInfo,
    Trajectory,
    create_stop_action,
)
from browser_env.actions import (
    ActionParsingError,
    create_id_based_action,
    create_none_action,
    create_playwright_action,
    is_equivalent,
)
from browser_env.auto_login import get_site_comb_from_filepath
from browser_env.helper_functions import (
    RenderHelper,
    get_action_description,
)
from evaluation_harness import evaluator_router
from llms import call_llm, lm_config

LOG_FOLDER = "log_files"
Path(LOG_FOLDER).mkdir(parents=True, exist_ok=True)
LOG_FILE_NAME = f"{LOG_FOLDER}/log_{time.strftime('%Y%m%d%H%M%S', time.localtime())}_{random.randint(0, 10000)}.log"

logger = logging.getLogger("logger")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger.addHandler(console_handler)

file_handler = logging.FileHandler(LOG_FILE_NAME)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)


def early_stop(
    trajectory: Trajectory, max_steps: int, thresholds: dict[str, int]
) -> tuple[bool, str]:
    """Check whether need to early stop"""
    num_steps = (len(trajectory) - 1) / 2
    if num_steps >= max_steps:
        return True, f"Reach max steps {max_steps}"

    last_k_actions: list[Action]
    action_seq: list[Action]

    k = thresholds["parsing_failure"]
    last_k_actions = trajectory[1::2][-k:]  # type: ignore[assignment]
    if len(last_k_actions) >= k:
        if all(
            [
                action["action_type"] == ActionTypes.NONE
                for action in last_k_actions
            ]
        ):
            return True, f"Failed to parse actions for {k} times"

    k = thresholds["repeating_action"]
    last_k_actions = trajectory[1::2][-k:]  # type: ignore[assignment]
    action_seq = trajectory[1::2]  # type: ignore[assignment]

    if len(action_seq) == 0:
        return False, ""

    last_action: Action = action_seq[-1]

    if last_action["action_type"] != ActionTypes.TYPE:
        if len(last_k_actions) >= k:
            if all(
                [
                    is_equivalent(action, last_action)
                    for action in last_k_actions
                ]
            ):
                return True, f"Same action for {k} times"
    else:
        if (
            sum([is_equivalent(action, last_action) for action in action_seq])
            >= k
        ):
            return True, f"Same typing action for {k} times"

    return False, ""


def extract_thought_and_action(response: str, prompt_constructor) -> tuple[str, str]:
    """Extract thought and action from ReACT response"""
    # Try to extract thought (everything before "Action:" or action phrase)
    answer_phrase = prompt_constructor.instruction["meta_data"].get(
        "answer_phrase", "In summary, the next action I will perform is"
    )
    action_splitter = prompt_constructor.instruction["meta_data"].get(
        "action_splitter", "```"
    )
    
    # Find the action using the existing extraction method
    try:
        action_str = prompt_constructor.extract_action(response)
    except Exception:
        action_str = ""
    
    # Extract thought (everything before the answer phrase)
    thought = ""
    if answer_phrase in response:
        thought = response.split(answer_phrase)[0].strip()
    elif "Action:" in response:
        thought = response.split("Action:")[0].strip()
    elif action_splitter in response:
        # Extract everything before the first action splitter
        parts = response.split(action_splitter)
        if len(parts) > 0:
            thought = parts[0].strip()
    else:
        # If no clear separator, use everything before action
        thought = response
    
    return thought, action_str


def test_react(
    args: argparse.Namespace,
    agent,
    config_file_list: list[str],
) -> None:
    """ReACT-style test loop with explicit Thought/Action/Observation"""
    scores = []
    max_steps = args.max_steps

    early_stop_thresholds = {
        "parsing_failure": args.parsing_failure_th,
        "repeating_action": args.repeating_action_failure_th,
    }

    env = ScriptBrowserEnv(
        headless=not args.render,
        slow_mo=args.slow_mo,
        observation_type=args.observation_type,
        current_viewport_only=args.current_viewport_only,
        viewport_size={
            "width": args.viewport_width,
            "height": args.viewport_height,
        },
        save_trace_enabled=args.save_trace_enabled,
        sleep_after_execution=args.sleep_after_execution,
    )

    for config_file in config_file_list:
        try:
            render_helper = RenderHelper(
                config_file, args.result_dir, args.action_set_tag
            )

            # get intent
            with open(config_file, encoding="utf-8") as f:
                _c = json.load(f)
                intent = _c["intent"]
                task_id = _c["task_id"]
                # automatically login
                if _c["storage_state"]:
                    cookie_file_name = os.path.basename(_c["storage_state"])
                    comb = get_site_comb_from_filepath(cookie_file_name)
                    temp_dir = tempfile.mkdtemp()
                    # subprocess to renew the cookie
                    subprocess.run(
                        [
                            "python",
                            "browser_env/auto_login.py",
                            "--auth_folder",
                            temp_dir,
                            "--site_list",
                            *comb,
                        ]
                    )
                    _c["storage_state"] = f"{temp_dir}/{cookie_file_name}"
                    assert os.path.exists(_c["storage_state"])
                    # update the config file
                    config_file = f"{temp_dir}/{os.path.basename(config_file)}"
                    with open(config_file, "w", encoding="utf-8") as f:
                        json.dump(_c, f)

            logger.info(f"[Config file]: {config_file}")
            logger.info(f"[Intent]: {intent}")

            agent.reset(config_file)
            trajectory: Trajectory = []
            obs, info = env.reset(options={"config_file": config_file})
            state_info: StateInfo = {"observation": obs, "info": info}
            trajectory.append(state_info)

            meta_data = {"action_history": ["None"]}
            react_history = []  # Store Thought/Action/Observation cycles
            
            # ReACT while loop
            while True:
                early_stop_flag, stop_info = early_stop(
                    trajectory, max_steps, early_stop_thresholds
                )

                if early_stop_flag:
                    action = create_stop_action(f"Early stop: {stop_info}")
                    trajectory.append(action)
                    break

                # THOUGHT and ACTION in one step
                try:
                    # Construct prompt using the same prompt constructor
                    prompt = agent.prompt_constructor.construct(
                        trajectory, intent, meta_data=meta_data
                    )
                    
                    # Call LLM to get Thought + Action
                    response = call_llm(agent.lm_config, prompt)
                    force_prefix = agent.prompt_constructor.instruction[
                        "meta_data"
                    ].get("force_prefix", "")
                    response = f"{force_prefix}{response}"
                    
                    # Extract thought and action
                    thought, action_str = extract_thought_and_action(
                        response, agent.prompt_constructor
                    )
                    
                    logger.info(f"[Thought]: {thought}")
                    logger.info(f"[Action]: {action_str}")
                    
                    # Parse action
                    n = 0
                    while True:
                        try:
                            parsed_response = agent.prompt_constructor.extract_action(
                                response
                            )
                            if agent.action_set_tag == "id_accessibility_tree":
                                action = create_id_based_action(parsed_response)
                            elif agent.action_set_tag == "playwright":
                                action = create_playwright_action(parsed_response)
                            else:
                                raise ValueError(
                                    f"Unknown action type {agent.action_set_tag}"
                                )
                            action["raw_prediction"] = response
                            break
                        except ActionParsingError as e:
                            n += 1
                            if n >= agent.lm_config.gen_config["max_retry"]:
                                action = create_none_action()
                                action["raw_prediction"] = response
                                break
                    
                except ValueError as e:
                    action = create_stop_action(f"ERROR: {str(e)}")

                trajectory.append(action)

                action_str_desc = get_action_description(
                    action,
                    state_info["info"]["observation_metadata"],
                    action_set_tag=args.action_set_tag,
                    prompt_constructor=agent.prompt_constructor
                    if isinstance(agent, PromptAgent)
                    else None,
                )
                render_helper.render(
                    action, state_info, meta_data, args.render_screenshot
                )
                meta_data["action_history"].append(action_str_desc)

                if action["action_type"] == ActionTypes.STOP:
                    break

                # OBSERVATION step
                obs, _, terminated, _, info = env.step(action)
                state_info = {"observation": obs, "info": info}
                trajectory.append(state_info)
                
                # Store ReACT cycle
                react_history.append({
                    "thought": thought,
                    "action": action_str_desc,
                    "observation": obs.get("text", "")[:200]  # Truncate for logging
                })
                logger.info(f"[Observation]: {obs.get('text', '')[:200]}...")

                if terminated:
                    trajectory.append(create_stop_action(""))
                    break

            evaluator = evaluator_router(config_file)
            score = evaluator(
                trajectory=trajectory,
                config_file=config_file,
                page=env.page,
                client=env.get_page_client(env.page),
            )

            scores.append(score)

            if score == 1:
                logger.info(f"[Result] (PASS) {config_file}")
            else:
                logger.info(f"[Result] (FAIL) {config_file}")

            if args.save_trace_enabled:
                env.save_trace(
                    Path(args.result_dir) / "traces" / f"{task_id}.zip"
                )

        except Exception as e:
            if "OpenAI" in str(type(e)):
                logger.info(f"[OpenAI Error] {repr(e)}")
            else:
                logger.info(f"[Unhandled Error] {repr(e)}]")
            import traceback

            with open(Path(args.result_dir) / "error.txt", "a", encoding="utf-8") as f:
                f.write(f"[Config file]: {config_file}\n")
                f.write(f"[Unhandled Error] {repr(e)}\n")
                f.write(traceback.format_exc())

        render_helper.close()

    env.close()
    logger.info(f"Average score: {sum(scores) / len(scores)}")


def prepare(args: argparse.Namespace) -> None:
    """Prepare result directory"""
    result_dir = args.result_dir
    if not result_dir:
        result_dir = (
            f"cache/results_{time.strftime('%Y%m%d%H%M%S', time.localtime())}"
        )
    if not Path(result_dir).exists():
        Path(result_dir).mkdir(parents=True, exist_ok=True)
        args.result_dir = result_dir
        logger.info(f"Create result dir: {result_dir}")

    if not (Path(result_dir) / "traces").exists():
        (Path(result_dir) / "traces").mkdir(parents=True)

    with open(os.path.join(result_dir, "log_files.txt"), "a+", encoding="utf-8") as f:
        f.write(f"{LOG_FILE_NAME}\n")


def get_unfinished(config_files: list[str], result_dir: str) -> list[str]:
    """Get unfinished config files"""
    result_files = glob.glob(f"{result_dir}/*.html")
    task_ids = [
        os.path.basename(f).split(".")[0].split("_")[1] for f in result_files
    ]
    unfinished_configs = []
    for config_file in config_files:
        task_id = os.path.basename(config_file).split(".")[0]
        if task_id not in task_ids:
            unfinished_configs.append(config_file)
    return unfinished_configs


def dump_config(args: argparse.Namespace) -> None:
    """Dump config to file"""
    config_file = Path(args.result_dir) / "config.json"
    if not config_file.exists():
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=4)
            logger.info(f"Dump config to {config_file}")


if __name__ == "__main__":
    import glob
    
    parser = argparse.ArgumentParser(
        description="ReACT-style agent evaluation"
    )
    parser.add_argument(
        "--render", action="store_true", help="Render the browser"
    )
    parser.add_argument(
        "--slow_mo",
        type=int,
        default=0,
        help="Slow down the browser by the specified amount",
    )
    parser.add_argument(
        "--action_set_tag", default="id_accessibility_tree", help="Action type"
    )
    parser.add_argument(
        "--observation_type",
        choices=["accessibility_tree", "html", "image"],
        default="accessibility_tree",
        help="Observation type",
    )
    parser.add_argument(
        "--current_viewport_only",
        action="store_true",
        help="Only use the current viewport for the observation",
    )
    parser.add_argument("--viewport_width", type=int, default=1280)
    parser.add_argument("--viewport_height", type=int, default=720)
    parser.add_argument("--save_trace_enabled", action="store_true")
    parser.add_argument("--sleep_after_execution", type=float, default=0.0)
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--agent_type", type=str, default="prompt")
    parser.add_argument(
        "--instruction_path",
        type=str,
        default="agent/prompts/jsons/p_cot_id_actree_2s.json",
    )
    parser.add_argument(
        "--parsing_failure_th",
        help="When consecutive parsing failure exceeds this threshold, the agent will stop",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--repeating_action_failure_th",
        help="When consecutive repeating action exceeds this threshold, the agent will stop",
        type=int,
        default=3,
    )
    parser.add_argument("--provider", type=str, default="openai")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-0613")
    parser.add_argument("--mode", type=str, default="chat")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--context_length", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=384)
    parser.add_argument("--stop_token", type=str, default=None)
    parser.add_argument(
        "--max_retry",
        type=int,
        help="max retry times to perform generations when parsing fails",
        default=1,
    )
    parser.add_argument(
        "--max_obs_length",
        type=int,
        help="when not zero, will truncate the observation to this length before feeding to the model",
        default=1920,
    )
    parser.add_argument(
        "--model_endpoint",
        help="huggingface model endpoint",
        type=str,
        default="",
    )
    parser.add_argument("--test_start_idx", type=int, default=0)
    parser.add_argument("--test_end_idx", type=int, default=1000)
    parser.add_argument("--result_dir", type=str, default="")
    parser.add_argument("--render_screenshot", action="store_true", default=True)
    
    args = parser.parse_args()
    
    # Check compatibility
    if (
        args.action_set_tag == "id_accessibility_tree"
        and args.observation_type != "accessibility_tree"
    ):
        raise ValueError(
            f"Action type {args.action_set_tag} is incompatible with the observation type {args.observation_type}"
        )
    
    args.sleep_after_execution = 2.0
    prepare(args)
    
    test_file_list = []
    st_idx = args.test_start_idx
    ed_idx = args.test_end_idx
    for i in range(st_idx, ed_idx):
        test_file_list.append(f"config_files/{i}.json")
    if "debug" not in args.result_dir:
        test_file_list = get_unfinished(test_file_list, args.result_dir)
    
    if len(test_file_list) == 0:
        logger.info("No task left to run")
    else:
        print(f"Total {len(test_file_list)} tasks left")
        args.render = False
        args.render_screenshot = True
        args.save_trace_enabled = True
        args.current_viewport_only = True
        dump_config(args)
        
        agent = construct_agent(args)
        test_react(args, agent, test_file_list)

