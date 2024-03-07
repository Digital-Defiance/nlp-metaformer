#!/bin/bash

ps aux | grep "[.]\/code tunnel" | awk '{print $2}' | xargs kill
