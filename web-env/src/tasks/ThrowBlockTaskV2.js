/**
 * ThrowBlockTask V2 - Calibrated for meaningful difficulty
 *
 * Changes from V1:
 * - Closer basket (4.0m vs 6.0m)
 * - Larger basket opening (1.5m vs 1.2m)
 * - Lower basket height (1.0m vs 1.5m)
 * - Lighter block for easier throws
 * - Auto-aim assist when throwing
 *
 * Target success rates:
 * - NRL-F at 1g: 30-50%
 * - RL agents at 1g after training: 75-90%
 */

import { BaseTask } from './BaseTask.js';

export class ThrowBlockTaskV2 extends BaseTask {
  constructor(world, config) {
    super(world, config);

    // V2 CALIBRATED PARAMETERS
    // Basket configuration - EASIER
    this.basketDistance = config.basketDistance ?? 4.0;        // Was 6.0
    this.basketDistanceVariance = config.basketDistanceVariance ?? 0.5; // Was 1.0
    this.basketHeight = config.basketHeight ?? 1.0;            // Was 1.5

    // Platform
    this.platformSize = 6.0;    // Smaller arena
    this.platformHeight = 0.5;

    // Block - LIGHTER
    this.blockSize = 0.35;      // Slightly smaller
    this.blockMass = 0.8;       // Was 2.0 - much lighter

    // Basket - BIGGER
    this.basketWidth = 1.5;     // Was 1.2
    this.basketDepth = 1.5;     // Was 1.2
    this.basketWallHeight = 0.6;
    this.basketWallThickness = 0.1;

    // Agent
    this.agentRadius = 0.4;
    this.agentHeight = 1.8;
    this.agentMass = 70;

    // State
    this.holdingBlock = false;
    this.isGrounded = false;

    // V2: Recalibrated throw strengths for 4m distance
    this.throwStrengths = {
      weak: 4.0,      // ~2m distance
      medium: 6.0,    // ~4m distance (optimal for basket)
      strong: 8.5     // ~6m distance
    };

    // V2: Fixed throw angle (45 degrees is optimal for range)
    this.throwAngle = 45 * Math.PI / 180;

    // V2: Pickup range increased
    this.pickupRange = 1.8;     // Was 1.5

    // Action mapping
    this.actions = [
      'forward', 'back', 'left', 'right',
      'pick', 'drop',
      'throw_weak', 'throw_medium', 'throw_strong',
      'idle'
    ];

    this.setup();
  }

  setup() {
    // Calculate actual basket distance with variance
    const actualDistance = this.basketDistance +
      this.randomInRange(-this.basketDistanceVariance, this.basketDistanceVariance);

    // Platform
    this.world.createBox('platform', {
      position: [0, -this.platformHeight / 2, 0],
      size: [this.platformSize, this.platformHeight, this.platformSize],
      mass: 0
    });

    // Agent - starts closer to block
    this.world.createCapsule('agent', {
      position: [-1.5, this.agentHeight / 2 + 0.1, 0],
      radius: this.agentRadius,
      height: this.agentHeight,
      mass: this.agentMass
    });

    // Block - right next to agent
    this.world.createBox('block', {
      position: [-0.8, this.blockSize / 2 + 0.1, 0],
      size: [this.blockSize, this.blockSize, this.blockSize],
      mass: this.blockMass
    });

    // Basket - create as multiple walls + floor
    const basketX = actualDistance;
    const basketY = this.basketHeight;

    // Basket floor
    this.world.createBox('basketFloor', {
      position: [basketX, basketY, 0],
      size: [this.basketWidth, this.basketWallThickness, this.basketDepth],
      mass: 0
    });

    // Basket walls (4 sides)
    const wallOffset = (this.basketWidth - this.basketWallThickness) / 2;
    const wallHeight = this.basketWallHeight;

    this.world.createBox('basketWallLeft', {
      position: [basketX - wallOffset, basketY + wallHeight / 2, 0],
      size: [this.basketWallThickness, wallHeight, this.basketDepth],
      mass: 0
    });

    this.world.createBox('basketWallRight', {
      position: [basketX + wallOffset, basketY + wallHeight / 2, 0],
      size: [this.basketWallThickness, wallHeight, this.basketDepth],
      mass: 0
    });

    this.world.createBox('basketWallFront', {
      position: [basketX, basketY + wallHeight / 2, -wallOffset],
      size: [this.basketWidth, wallHeight, this.basketWallThickness],
      mass: 0
    });

    this.world.createBox('basketWallBack', {
      position: [basketX, basketY + wallHeight / 2, wallOffset],
      size: [this.basketWidth, wallHeight, this.basketWallThickness],
      mass: 0
    });

    // Store basket bounds for detection
    this.basketBounds = {
      minX: basketX - this.basketWidth / 2 + this.basketWallThickness,
      maxX: basketX + this.basketWidth / 2 - this.basketWallThickness,
      minY: basketY,
      maxY: basketY + this.basketWallHeight,
      minZ: -this.basketDepth / 2 + this.basketWallThickness,
      maxZ: this.basketDepth / 2 - this.basketWallThickness
    };

    this.actualBasketDistance = actualDistance;
  }

  getObservation() {
    const agentState = this.world.getBodyState('agent');
    const blockState = this.world.getBodyState('block');

    // V2: Calculate distance to basket for agent
    const distToBasket = this.actualBasketDistance - agentState.position[0];

    return {
      // Agent state
      agentPosition: agentState.position,
      agentVelocity: agentState.velocity,

      // Block state
      blockPosition: blockState.position,
      blockVelocity: blockState.velocity,
      holdingBlock: this.holdingBlock,

      // Basket info
      basketPosition: [this.actualBasketDistance, this.basketHeight, 0],
      basketBounds: this.basketBounds,
      distanceToBasket: distToBasket,

      // V2: Throw guidance
      throwStrengths: this.throwStrengths,
      optimalThrowStrength: this._getOptimalThrowStrength(distToBasket),

      // Physics info
      gravity: this.config.gravity,
      isGrounded: this.isGrounded,

      // Available actions
      actions: this.actions
    };
  }

  _getOptimalThrowStrength(distance) {
    // Simple heuristic: which throw strength is closest to needed?
    // Range formula: R = v^2 * sin(2*theta) / g
    const g = this.config.gravity;
    const theta = this.throwAngle;

    let bestStrength = 'medium';
    let minError = Infinity;

    for (const [name, v] of Object.entries(this.throwStrengths)) {
      const range = (v * v * Math.sin(2 * theta)) / g;
      const error = Math.abs(range - distance);
      if (error < minError) {
        minError = error;
        bestStrength = name;
      }
    }

    return bestStrength;
  }

  applyAction(action, params = {}) {
    if (params.durationScale && params.durationScale !== 1.0) {
      console.log(`Applying variable granularity scale: ${params.durationScale}`);
    }
    const agent = this.world.getBody('agent');
    const block = this.world.getBody('block');
    if (!agent || !block) return;

    // Check if grounded
    this.isGrounded = Math.abs(agent.velocity.y) < 0.1 && agent.position.y < this.agentHeight / 2 + 0.3;

    // Normalize action
    const actionStr = typeof action === 'number' ? this.actions[action] : action.toLowerCase();

    const scale = params.durationScale || 1.0;
    switch (actionStr) {
      case 'forward':
        agent.velocity.x = 3.0 * scale;
        break;

      case 'back':
        agent.velocity.x = -3.0 * scale;
        break;

      case 'left':
        agent.velocity.z = -3.0 * scale;
        break;

      case 'right':
        agent.velocity.z = 3.0 * scale;
        break;

      case 'pick':
        if (!this.holdingBlock) {
          // Check if close enough to block
          const dx = agent.position.x - block.position.x;
          const dz = agent.position.z - block.position.z;
          const dist = Math.sqrt(dx * dx + dz * dz);

          if (dist < this.pickupRange) {
            this.holdingBlock = true;
          }
        }
        break;

      case 'drop':
        if (this.holdingBlock) {
          this.holdingBlock = false;
          block.position.set(
            agent.position.x + 0.5,
            agent.position.y - 0.5,
            agent.position.z
          );
          block.velocity.set(0, 0, 0);
        }
        break;

      case 'throw_weak':
      case 'throw_medium':
      case 'throw_strong':
        if (this.holdingBlock) {
          this.holdingBlock = false;

          // Get throw strength
          const strengthKey = actionStr.replace('throw_', '');
          const strength = this.throwStrengths[strengthKey];

          // Position block in front of agent
          block.position.set(
            agent.position.x + 0.6,
            agent.position.y + 0.4,
            agent.position.z
          );

          // V2: Use fixed optimal angle and auto-aim
          // Calculate direction to basket center
          const toBasketX = this.actualBasketDistance - block.position.x;
          const toBasketZ = 0 - block.position.z;
          const horizontalDist = Math.sqrt(toBasketX * toBasketX + toBasketZ * toBasketZ);

          // Normalize direction
          const dirX = toBasketX / horizontalDist;
          const dirZ = toBasketZ / horizontalDist;

          // V2: Apply throw with optimal angle
          const vHorizontal = strength * Math.cos(this.throwAngle);
          const vVertical = strength * Math.sin(this.throwAngle);

          block.velocity.set(
            vHorizontal * dirX,
            vVertical,
            vHorizontal * dirZ
          );
        }
        break;

      case 'idle':
      default:
        agent.velocity.x *= 0.9;
        agent.velocity.z *= 0.9;
        break;
    }

    // If holding block, move block with agent
    if (this.holdingBlock) {
      block.position.set(
        agent.position.x + 0.5,
        agent.position.y + 0.2,
        agent.position.z
      );
      block.velocity.set(0, 0, 0);
    }

    // Clamp agent velocity
    const maxSpeed = 4.0;
    agent.velocity.x = Math.max(-maxSpeed, Math.min(maxSpeed, agent.velocity.x));
    agent.velocity.z = Math.max(-maxSpeed, Math.min(maxSpeed, agent.velocity.z));
  }

  checkTermination() {
    const block = this.world.getBody('block');
    const agent = this.world.getBody('agent');

    if (!block || !agent) {
      return { done: true, success: false, reason: 'missing_objects' };
    }

    // Check if block fell off platform
    if (block.position.y < -2) {
      return { done: true, success: false, reason: 'block_fell' };
    }

    // Check if agent fell off platform
    if (agent.position.y < -2) {
      return { done: true, success: false, reason: 'agent_fell' };
    }

    // Check if block is in basket and at rest
    const pos = block.position;
    const vel = block.velocity;
    const speed = Math.sqrt(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z);

    const inBasket =
      pos.x >= this.basketBounds.minX && pos.x <= this.basketBounds.maxX &&
      pos.y >= this.basketBounds.minY && pos.y <= this.basketBounds.maxY &&
      pos.z >= this.basketBounds.minZ && pos.z <= this.basketBounds.maxZ;

    const atRest = speed < 0.5;

    if (inBasket && atRest) {
      this.successSteps++;
      if (this.successSteps >= this.requiredSuccessSteps) {
        return { done: true, success: true, reason: 'block_in_basket' };
      }
    } else {
      this.successSteps = 0;
    }

    return { done: false, success: false, reason: 'ongoing' };
  }

  getReward(done, success) {
    return success ? 1.0 : 0.0;
  }

  // V2: Physics calculation helper
  getPhysicsInfo() {
    const g = this.config.gravity;
    const theta = this.throwAngle;

    const ranges = {};
    for (const [name, v] of Object.entries(this.throwStrengths)) {
      ranges[name] = (v * v * Math.sin(2 * theta)) / g;
    }

    return {
      gravity: g,
      throwAngle: theta * 180 / Math.PI,
      throwRanges: ranges,
      basketDistance: this.actualBasketDistance,
      basketHeight: this.basketHeight
    };
  }
}
