"""
PyEventBT
Documentation: https://pyeventbt.com
GitHub: https://github.com/marticastany/pyeventbt

Author: Marti Castany
Copyright (c) 2025 Marti Castany
Licensed under the Apache License, Version 2.0
"""

from typing import Callable, Dict, List

from pyeventbt.events.events import BarEvent, ScheduledEvent
from pyeventbt.strategy.core.modules import Modules
from pyeventbt.strategy.core.strategy_timeframes import StrategyTimeframes
from pydantic import BaseModel
import pandas as pd

class Schedule(BaseModel):
    name: str
    is_active: bool = True
    fn: Callable
    execute_every: StrategyTimeframes
    
    def __eq__(self, value: object) -> bool:
        return self.name == value.name

class Schedules:
    
    def __init__(self) -> None:
        self.__schedules: Dict[StrategyTimeframes, List[Schedule]] = {}

    def add_schedule(self, timeframe: StrategyTimeframes, callback: Callable[[ScheduledEvent, Modules], None]) -> Schedule:
        schedule = Schedule(
            name=repr(callback),
            is_active=True,
            fn=callback,
            execute_every=timeframe
        )
        self.__schedules.setdefault(timeframe, []).append(schedule)
        return schedule
    
    def activate_schedule(self, schedule: Schedule):
        schedules = self.__schedules[schedule.execute_every]
        
        for sc in schedules:
            if sc == schedule:
                sc.is_active = True
    
    def deactivate_schedule(self, schedule: Schedule):
        schedules = self.__schedules[schedule.execute_every]
        
        for sc in schedules:
            if sc == schedule:
                sc.is_active = False
    
    def deactivate_all_schedules(self):
        
        for _, schedules in self.__schedules.items():
            for schedule in schedules:
                schedule.is_active = False
    
    def activate_all_schedules(self):
        for _, schedules in self.__schedules.items():
            for schedule in schedules:
                schedule.is_active = True
    
    def remove_schedule(self, schedule: Schedule) -> None:
        
        res = self.__schedules.pop(schedule, None)
        if res is None:
            print(f"WARNING: no schedule found to remove that matches {schedule}")
    
    def remove_inactive_schedules(self) -> None:
        
            self.__schedules = {key: self.__schedules[key] for key in self.__schedules.keys() if key.is_active == True}
    
    def get_callbacks_to_execute_given_timeframe(self, timeframe: StrategyTimeframes) -> List[Callable[[ScheduledEvent, Modules], None]]:
        
        schedules = self.__schedules.get(timeframe, [])
        
        return [schedule.fn for schedule in schedules if schedule.is_active]

class TimeframeWatchInfo(BaseModel):
    last_timestamp: pd.Timestamp = None
    current_timestamp: pd.Timestamp = None

    class Config:
        arbitrary_types_allowed = True
        
    def __eq__(self, value: object) -> bool:
        if isinstance(value, TimeframeWatchInfo):
            return self.last_timestamp == value.last_timestamp and self.current_timestamp == value.current_timestamp
        else:
            ValueError(f"Cannot compare TimeframeWatchInfo and {type(value)}")

class ScheduleService:
    
    
    def __init__(self, modules: Modules) -> None:
        """Executes callbacks giving a configuration
        """
        self.__modules = modules
        self.__schedules = Schedules()
        self.__timeframes_to_watch: Dict[StrategyTimeframes, TimeframeWatchInfo] = {}
        self.__last_callback_args: Dict[str, ScheduledEvent] = {}
    
    def __get_timeframes_to_trigger(self, event: BarEvent) -> List[StrategyTimeframes]:
        
        timeframes_to_trigger = []
        
        # OLD: if event.data.empty:
        # NEW: Instead check if we have valid data by looking for the timestamp
        if not hasattr(event, "datetime") or event.datetime is None:
            return []
        
        for k, v in self.__timeframes_to_watch.items():
            
            # check if current_timestamp is None
            if v.current_timestamp is None:
                # assign the same timestamp to current timestamp and last_timestamp
                v.current_timestamp = event.datetime #data.name
                v.last_timestamp = event.datetime #data.name
                
                continue
            else:
                v.current_timestamp = event.datetime #data.name
                # check if the difference between 
                # last timestamp and current timestamp is bigger than the interval
                if v.current_timestamp - v.last_timestamp >= k.to_timedelta():
                    timeframes_to_trigger.append(k)
        
        return timeframes_to_trigger
    
    
    def add_schedule(self, timeframe: StrategyTimeframes, callback: Callable[[ScheduledEvent, Modules], None]):
        sc = self.__schedules.add_schedule(
            timeframe=timeframe,
            callback=callback
        )
        self.__timeframes_to_watch.setdefault(timeframe, TimeframeWatchInfo())
        return sc
        
    def deactivate_schedules(self):
        self.__schedules.deactivate_all_schedules()
        
    def activate_schedules(self):
        self.__schedules.activate_all_schedules()
    
    def run_scheduled_callbacks(self, event: BarEvent):
        
        timeframes_to_trigger = self.__get_timeframes_to_trigger(event)
        
        for timeframe in timeframes_to_trigger:
            
            timeframe_watch = self.__timeframes_to_watch[timeframe]
            
            for cb in self.__schedules.get_callbacks_to_execute_given_timeframe(timeframe):
                
                # check that the last 
                
                cb(
                    ScheduledEvent(
                        schedule_timeframe=timeframe,
                        symbol=event.symbol,
                        timestamp=timeframe_watch.current_timestamp,
                        former_execution_timestamp=timeframe_watch.last_timestamp,
                    ),
                    self.__modules
                )

                # once callback has been executed update timesmps
                timeframe_watch.last_timestamp = timeframe_watch.current_timestamp 