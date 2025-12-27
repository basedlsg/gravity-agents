/**
 * BaseTask - Abstract base class for physics tasks
 */

export class BaseTask {
  constructor(world, config) {
    this.world = world;
    this.config = config;
    this.successSteps = 0;
    this.requiredSuccessSteps = 5;
  }

  /**
   * Set up the task scene (platforms, objects, etc.)
   * Override in subclasses
   */
  setup() {
    throw new Error('setup() must be implemented');
  }

  /**
   * Get current observation
   * Override in subclasses
   */
  getObservation() {
    throw new Error('getObservation() must be implemented');
  }

  /**
   * Apply an action
   * Override in subclasses
   */
  applyAction(action) {
    throw new Error('applyAction() must be implemented');
  }

  /**
   * Check if task is done
   * Override in subclasses
   * @returns {{ done: boolean, success: boolean, reason: string }}
   */
  checkTermination() {
    throw new Error('checkTermination() must be implemented');
  }

  /**
   * Get reward for current state
   * Override in subclasses for shaped rewards
   */
  getReward(done, success) {
    // Sparse reward by default
    return success ? 1.0 : 0.0;
  }

  /**
   * Simple seeded random number generator
   */
  seededRandom() {
    // LCG parameters
    this.config.seed = (this.config.seed * 1103515245 + 12345) & 0x7fffffff;
    return (this.config.seed / 0x7fffffff);
  }

  /**
   * Get a random value in range
   */
  randomInRange(min, max) {
    return min + this.seededRandom() * (max - min);
  }
}
