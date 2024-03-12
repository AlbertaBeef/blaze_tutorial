#!/bin/bash
set -e

RESUME_CONTAINER=false
OVERRIDE_CONTAINER=false

readonly CONTAINER_NAME="hailo_ai_sw_suite_2023-10_container"
readonly XAUTH_FILE=/tmp/hailo_docker.xauth
readonly DOCKER_TAR_FILE="hailo_ai_sw_suite_2023-10.tar"
readonly DOCKER_IMAGE_NAME="hailo_ai_sw_suite_2023-10:1"
readonly NVIDIA_GPU_EXIST=$(lspci | grep "VGA compatible controller: NVIDIA")
readonly NVIDIA_DOCKER_EXIST=$(dpkg -l | grep nvidia-docker)
readonly SHARED_DIR="shared_with_docker"
readonly DEFAULT_HAILORT_LOGGER_PATH="/var/log/hailo"

readonly WHITE="\e[0m"
readonly CYAN="\e[1;36m"
readonly RED="\e[1;31m"
readonly YELLOW="\e[0;33m"

#
# System requirements' check
#

readonly table_file="system_reqs_table.log"
readonly log_file="system_reqs_results.log"
readonly log_boundary=" | "
readonly log_found="V$log_boundary"   # V marks a satisfied requirement.
readonly log_missing="X$log_boundary" # X marks an unsatisfied requirement.
readonly log_warning=" $log_boundary"

# DFC requirements:
readonly req_mem=16
readonly rec_mem=32
readonly req_arch="x86_64"
readonly req_driver=525

declare -a cpu_commands
cpu_commands[0]='avx;0'

declare -a cpu_commands_reasons
cpu_commands_reasons[0]='install TensorFlow'
# End of DFC requirements

error=false

function check_ram() {
    # Get total RAM size in GB:
    local mem=$(free -g -t | grep Total | awk '{print $2}')

    if [ $req_mem -gt $mem ]; then
        error=true
        echo "ERROR: The Dataflow Compiler requires $req_mem GB of RAM ($rec_mem GB recommended), while this system has only $mem GB." 1>&2
        echo "$log_missing Insufficient RAM: $req_mem GB of RAM are required, only $mem GB available." >> $log_file
    else
        if [ $rec_mem -gt $mem ]; then
            echo "WARNING: It is recommended to have $rec_mem GB of RAM, while this system has only $mem GB." 1>&2
            echo "$log_warning Available RAM ($mem GB) below recommended amount ($rec_mem GB)." >> $log_file
        else
            echo "$log_found Available RAM ($mem GB) is sufficient, and within recommendation ($rec_mem GB)." >> $log_file
        fi
    fi
    echo -e "RAM(GB)\t${req_mem}\t${mem}\tRequired" >> $table_file
    echo -e "RAM(GB)\t${rec_mem}\t${mem}\tRecommended" >> $table_file
}

function check_cpu() {
    # Get CPU architecture:
    local cpu_arch=$(lscpu | grep Architecture | awk '{print $2}')
    if [ $cpu_arch != $req_arch ]; then
        error=true
        echo "ERROR: CPU architecture required is $req_arch, found $cpu_arch." 1>&2
        echo "$log_missing Unsupported CPU architecture: $cpu_arch. The supported architecture is $req_arch." >> $log_file
    else
        echo "$log_found CPU architecture $cpu_arch is supported." >> $log_file
    fi
    echo -e "CPU-Arch\t${req_arch}\t${cpu_arch}\tRequired" >> $table_file
}

function check_gpu() {
    nvidia-smi >>/dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "INFO: No GPU connected."
        echo "$log_warning GPU Requirements are not checked- no GPU connected." >> $log_file
    else
        # Get GPU details:
        local driver_ver=$(nvidia-smi -q | grep "Driver Version" | awk '{print $4}')
        local driver_ver=${driver_ver:0:3}

        # Compare to requirements:
        if [ $req_driver -gt $driver_ver ]; then
            echo "WARNING: GPU driver version should be $req_driver or higher, found $driver_ver." 1>&2
            echo "$log_missing GPU driver version should be $req_driver or higher, found $driver_ver." >> $log_file
        else
            echo "$log_found GPU driver version is $driver_ver." >> $log_file
        fi
        echo -e "GPU-Driver\t${req_driver}\t${driver_ver}\tRecommended" >> $table_file
    fi
}

function check_cpu_instructions() {
    for opcode in ${cpu_commands[@]}
    do
        IFS=";" read -r -a arr <<< "${opcode}"
        local command_name="${arr[0]}"
        local reason=${cpu_commands_reasons["${arr[1]}"]}
        lscpu | grep $command_name > /dev/null
        if [ "$?" != "0" ]; then
            error=true
            echo "ERROR: CPU flag $command_name is not supported in this CPU, and is required to $reason." 1>&2
            echo "$log_missing Required $command_name CPU flag is not supported in this CPU, and is required to $reason." >> $log_file
            local found="X"
        else
            echo "$log_found Required $command_name CPU flag is supported." >> $log_file
            local found="V"
        fi
        echo -e "CPU-flag\t${command_name}\t${found}\tRequired" >> $table_file
    done
}

function print_report() {
    echo -e "\nSYSTEM REQUIREMENTS REPORT\n"
    column -t $table_file
    echo ""
    echo "See $log_file for more information."
    rm -f $table_file
    exit 1
}

function prepare_log_file() {
    rm -f $table_file
    rm -f $log_file

    echo -e "Component\tRequirement\tFound" >> $table_file
    echo -e "==========\t==========\t==========\t========== " >> $table_file
    echo -e "HAILO System requirements check - log\n" >> $log_file
}

function check_result() {
    if [ $error == true ]; then
        echo "ERROR: System requirements check failed."
        print_report
    else
        echo "INFO: System requirements check finished successfully."
    fi
    rm $table_file
}

function check_system_requirements_dfc() {
    check_ram
    check_cpu
    check_cpu_instructions
    check_gpu
}

function check_system_requirements() {
    echo "INFO: Checking system requirements..."
    prepare_log_file
    check_system_requirements_dfc
    check_result
}

#
# End of system requirements' check
#

function print_usage() {
    echo "Running Hailo AI Software Suite Docker image:"
    echo "The default mode will create a new container. If one already exists, use --resume / --override"
    echo ""
    echo "  -h, --help                                 Show help"
    echo "  --resume                                   Resume the old container"
    echo "  --override                                 Delete the existing container and start a new one"
    echo "  --hailort-enable-service                   Run HailoRT service"
    echo ""
    echo "  Options for HailoRT service:"
    echo "  --service-enable-monitor"
    echo "  --service-hailort-logger-path </path>      Note: the path is set inside docker container"
    exit 1
}

function parse_args() {
    while test $# -gt 0; do
        if [[ "$1" == "-h" || "$1" == "--help" ]]; then
            print_usage
        elif [ "$1" == "--resume" ]; then
            RESUME_CONTAINER=true
        elif [ "$1" == "--override" ]; then
            OVERRIDE_CONTAINER=true
        elif [ "$1" == "--hailort-enable-service" ]; then
            HAILORT_ENABLE_SERVICE=true
        elif [ "$1" == "--service-enable-monitor" ]; then
            ENABLE_HAILO_MONITOR=true
        elif [ "$1" == "--service-hailort-logger-path" ]; then
            HAILORT_LOGGER_PATH="$2"
            shift
        else
            echo "Unknown option: $1" && exit 1
        fi
	shift
    done
}

function prepare_docker_args() {
    DOCKER_ARGS="--privileged \
                 --net=host \
                 -e DISPLAY=$DISPLAY \
                 -e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
                 --device=/dev/dri:/dev/dri \
                 --ipc=host \
                 --group-add 44 \
                 -v /dev:/dev \
                 -v /lib/firmware:/lib/firmware \
                 -v /lib/modules:/lib/modules \
                 -v /lib/udev/rules.d:/lib/udev/rules.d \
                 -v /usr/src:/usr/src \
                 -v ${XAUTH_FILE}:/home/hailo/.Xauthority \
                 -v /tmp/.X11-unix/:/tmp/.X11-unix/ \
                 --name $CONTAINER_NAME \
                 -v /var/run/docker.sock:/var/run/docker.sock \
                 -v /etc/machine-id:/etc/machine-id:ro \
                 -v /var/run/dbus/system_bus_socket:/var/run/dbus/system_bus_socket \
                 -v $(pwd)/${SHARED_DIR}/:/local/${SHARED_DIR}:rw \
                 -v /etc/timezone:/etc/timezone:ro \
                 -v /etc/localtime:/etc/localtime:ro
                "
    if [[ -d "/var/lib/dkms" ]]; then
         DOCKER_ARGS+="-v /var/lib/dkms:/var/lib/dkms "
    fi
    if [[ $NVIDIA_GPU_EXIST ]] && [[ $NVIDIA_DOCKER_EXIST ]]; then
        DOCKER_ARGS+="--gpus all "
    fi
    if [[ $HAILORT_ENABLE_SERVICE ]]; then
        DOCKER_ARGS+="-e hailort_enable_service=yes "
        if [[ $ENABLE_HAILO_MONITOR ]]; then
            DOCKER_ARGS+="-e HAILO_MONITOR=1 "
        fi
        if [[ $HAILORT_LOGGER_PATH ]]; then
            DOCKER_ARGS+="-e HAILORT_LOGGER_PATH=${HAILORT_LOGGER_PATH} "
        else
            DOCKER_ARGS+="-e HAILORT_LOGGER_PATH=${DEFAULT_HAILORT_LOGGER_PATH} "
        fi
    fi
}

function load_hailo_ai_sw_suite_image() {
    SCRIPT_DIR=$(realpath $(dirname ${BASH_SOURCE[0]}))
    DOCKER_FILE_PATH="${SCRIPT_DIR}/${DOCKER_TAR_FILE}"
    if [[ ! -f "${DOCKER_FILE_PATH}" ]]; then
        echo -e "${RED}Missing file: $DOCKER_FILE_PATH${WHITE} " && exit 1
    fi
    echo -e "${CYAN}Loading Docker image: $DOCKER_FILE_PATH${WHITE}"
    set +e && check_system_requirements && set -e
    docker load -i $DOCKER_FILE_PATH
}

function run_hailo_ai_sw_suite_image() {
    prepare_docker_args
    RUN_CMD="docker run ${DOCKER_ARGS} -ti $1"
    echo -e "${CYAN}Running Hailo AI SW suite Docker image with the folowing Docker command:${WHITE}" && echo $RUN_CMD
    $RUN_CMD
}

function check_docker_install_and_user_permmision() {
    if [[ ! $(which docker) ]]; then
        echo -e "${RED}Docker is not installed${WHITE}" && exit 1 
    fi
    docker images &> /dev/null
    if [[ $? -ne 0 ]]; then
        echo -e "${RED}The current user:${USER} is not in the 'Docker' group${WHITE}" && exit 1
    fi
}

function run_new_container() {
    check_docker_install_and_user_permmision
    if [[ $NUM_OF_CONTAINERS_EXSISTS -ge 1 ]]; then
        echo -e "${RED}Can't start a new container, already found one. Consider using --resume or --override${WHITE}"
        echo -e "${RED}In case of replacing the Hailo AI SW Suite image, delete the existing containers and images${WHITE}"
        echo -e "${RED}Caution, all data from the exist container will be erased. To prevent data loss, save it to your own Docker volume${WHITE}" && exit 1
    elif [ "$(docker images -q $DOCKER_IMAGE_NAME 2> /dev/null)" == "" ]; then
        load_hailo_ai_sw_suite_image
    fi
    echo -e "${CYAN}Starting new container${WHITE}"
    run_hailo_ai_sw_suite_image $DOCKER_IMAGE_NAME
}

function overide_container() {
    if [[ "$NUM_OF_CONTAINERS_EXSISTS" -ge "1" ]]  ; then
        echo -e "${CYAN}Overriding old container${WHITE}"
        docker stop "$CONTAINER_NAME" > /dev/null
        docker rm "$CONTAINER_NAME" > /dev/null
	NUM_OF_CONTAINERS_EXSISTS=$(docker ps -a -q -f "name=$CONTAINER_NAME" | wc -l)
    fi
    run_new_container
}

function resume_container() {
    if [[ "$NUM_OF_CONTAINERS_EXSISTS" -lt "1" ]]; then
        echo -e "${RED}Found no container. please run for the first time without --resume${WHITE} $1"
        exit 1
    fi

    echo -e "${CYAN}Resuming an old container${WHITE} $1"
    # Start and then exec in order to pass the DISPLAY env, because this vairble
    # might change from run to run (after reboot for example)
    docker start "$CONTAINER_NAME"
    local DOCKER_RESUME_ARGS="-e DISPLAY=$DISPLAY "
    if [[ $HAILORT_ENABLE_SERVICE ]]; then
        DOCKER_RESUME_ARGS+="-e hailort_enable_service=yes "
        if [[ $ENABLE_SCHEDULER_MONITOR ]]; then
            DOCKER_RESUME_ARGS+="-e SCHEDULER_MONITOR=1 "
        fi
        if [[ $HAILORT_LOGGER_PATH ]]; then
            DOCKER_RESUME_ARGS+="-e HAILORT_LOGGER_PATH=${HAILORT_LOGGER_PATH} "
        fi
    fi
    docker exec -ti $DOCKER_RESUME_ARGS "$CONTAINER_NAME" /bin/bash
}

function create_shared_dir() {
    mkdir -p ${SHARED_DIR}
    chmod 777 ${SHARED_DIR}
}

function handle_xauth() {
    [[ -d $XAUTH_FILE ]] && print_xauth_file_error
    touch $XAUTH_FILE
    xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH_FILE nmerge -
    chmod o+rw $XAUTH_FILE
}

function print_xauth_file_error(){
cat <<  EOF

    Error:   It looks like there was an attempt to start the "Hailo AI SW Suite" container with means
             other than running this script. Unfortunately, this is not fully supported.
             Please run the command below before attempting to start the "Hailo AI SW Suite" container:
             
             sudo rm -r ${XAUTH_FILE}

             And then run this script again to start or resume the container.

EOF
    exit 1
}

function main() {
    parse_args "$@"
    create_shared_dir
    # Critical for display
    xhost + &> /dev/null || true
    handle_xauth
    NUM_OF_CONTAINERS_EXSISTS=$(docker ps -a -q -f "name=$CONTAINER_NAME" | wc -l)
    if [ "$RESUME_CONTAINER" = true ]; then
        resume_container
    elif [ "$OVERRIDE_CONTAINER" = true ]; then
        overide_container
    else
        run_new_container
    fi
}

main "$@"
