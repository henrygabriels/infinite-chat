#!/bin/bash

# LLM Chat Client Launcher
# Starts the server if needed and opens the terminal client

set -e

# Ensure we're in the correct directory
cd "$(dirname "$0")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Server configuration
SERVER_HOST="localhost"
SERVER_PORT="8421"
SERVER_URL="http://${SERVER_HOST}:${SERVER_PORT}"
HEALTH_ENDPOINT="${SERVER_URL}/api/health"

echo -e "${BLUE}🚀 infinite chat Launcher${NC}"
echo ""

# Function to check if server is running
check_server() {
    if curl -s "$HEALTH_ENDPOINT" > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Function to start server
start_server() {
    echo -e "${YELLOW}Starting LLM server...${NC}"

    # Start server in background
    uv run python main.py > /tmp/llm_server.log 2>&1 &
    SERVER_PID=$!

    echo "Server PID: $SERVER_PID"

    # Wait for server to be ready
    echo -n "Waiting for server to start"
    for i in {1..30}; do
        if check_server; then
            echo -e "\n${GREEN}✓ Server is ready!${NC}"
            return 0
        fi
        echo -n "."
        sleep 1
    done

    echo -e "\n${RED}✗ Server failed to start within 30 seconds${NC}"
    echo "Check the log: /tmp/llm_server.log"
    return 1
}

# Main execution
main() {
    # Check if server is already running
    if check_server; then
        echo -e "${GREEN}✓ Server is already running at $SERVER_URL${NC}"
    else
        echo -e "${YELLOW}Server is not running. Starting it now...${NC}"
        if ! start_server; then
            exit 1
        fi
    fi

    echo ""
    echo -e "${BLUE}📱 Starting terminal client...${NC}"
    echo -e "${YELLOW}Press Ctrl+C to exit the client${NC}"
    echo ""

    # Start the client
    uv run python client.py
}

# Trap to handle script interruption
trap 'echo -e "\n${YELLOW}Shutting down...${NC}"; exit 0' INT

# Run main function
main "$@"