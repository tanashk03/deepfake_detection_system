#!/bin/bash
# ============================================
# Luminark Local Development Runner
# One command to bring up the entire stack
# ============================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}"
echo "╔═══════════════════════════════════════════╗"
echo "║     Luminark Local Development Stack      ║"
echo "╚═══════════════════════════════════════════╝"
echo -e "${NC}"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    exit 1
fi

if ! docker info &> /dev/null; then
    echo -e "${RED}Error: Docker daemon is not running${NC}"
    exit 1
fi

# Parse arguments
ACTION="${1:-up}"

case "$ACTION" in
    up|start)
        echo -e "${YELLOW}Starting Luminark stack...${NC}"
        cd "$PROJECT_ROOT"
        
        # Build and start
        docker compose up --build -d
        
        echo ""
        echo -e "${GREEN}✓ Stack started successfully!${NC}"
        echo ""
        echo "Services:"
        echo "  • Frontend:  http://localhost:3000"
        echo "  • Backend:   http://localhost:8000"
        echo "  • API Docs:  http://localhost:8000/docs"
        echo "  • Database:  localhost:5432"
        echo "  • Redis:     localhost:6379"
        echo ""
        echo "Logs: docker compose logs -f"
        echo "Stop: $0 down"
        ;;
        
    down|stop)
        echo -e "${YELLOW}Stopping Luminark stack...${NC}"
        cd "$PROJECT_ROOT"
        docker compose down
        echo -e "${GREEN}✓ Stack stopped${NC}"
        ;;
        
    restart)
        echo -e "${YELLOW}Restarting Luminark stack...${NC}"
        cd "$PROJECT_ROOT"
        docker compose restart
        echo -e "${GREEN}✓ Stack restarted${NC}"
        ;;
        
    logs)
        cd "$PROJECT_ROOT"
        docker compose logs -f "${@:2}"
        ;;
        
    status)
        cd "$PROJECT_ROOT"
        docker compose ps
        ;;
        
    clean)
        echo -e "${YELLOW}Cleaning up volumes and images...${NC}"
        cd "$PROJECT_ROOT"
        docker compose down -v --rmi local
        echo -e "${GREEN}✓ Cleanup complete${NC}"
        ;;
        
    build)
        echo -e "${YELLOW}Building images...${NC}"
        cd "$PROJECT_ROOT"
        docker compose build "${@:2}"
        echo -e "${GREEN}✓ Build complete${NC}"
        ;;
        
    *)
        echo "Usage: $0 {up|down|restart|logs|status|clean|build}"
        echo ""
        echo "Commands:"
        echo "  up       Start the development stack"
        echo "  down     Stop the stack"
        echo "  restart  Restart all services"
        echo "  logs     Follow service logs"
        echo "  status   Show service status"
        echo "  clean    Remove volumes and images"
        echo "  build    Rebuild images"
        exit 1
        ;;
esac
